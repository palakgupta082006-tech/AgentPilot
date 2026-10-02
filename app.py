"""
AgentPilot
AI Coding Agent & Engineering Simulator
Flask + Gemini tool-using coding agent

Run:
    python app.py

Open:
    http://127.0.0.1:5000
"""

import os
import shlex
import shutil
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path

from dotenv import load_dotenv
from flask import Flask, jsonify, request, send_from_directory
from google import genai
from google.genai import types


# ============================================================
# CONFIG
# ============================================================

load_dotenv()

MODEL = os.environ.get(
    "AGENTPILOT_MODEL",
    "gemini-3.8-flash"
)

MAX_ITERATIONS = 30
CHECKPOINT_TIMEOUT = 600

ROOT = Path(__file__).parent.resolve()
WORKSPACES = ROOT / "workspaces"
SAMPLES = ROOT / "sample_repos"

SAFE_COMMANDS = {
    "python",
    "python3",
    "pytest",
    "ls",
    "cat",
}

app = Flask(
    __name__,
    static_folder=str(ROOT),
    static_url_path=""
)

MISSIONS = {}


# ============================================================
# GEMINI CLIENT
# ============================================================

api_key = os.environ.get("GEMINI_API_KEY")

client = None

if api_key:

    client = genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(
            timeout=30000
        )
    )


# ============================================================
# AGENT SYSTEM PROMPT
# ============================================================

SYSTEM = """
You are AgentPilot's autonomous coding agent.

A human pilot has given you an engineering mission.

You work inside a workspace using tools.

Your job is to behave like a senior software engineer.

WORKFLOW:

1. Inspect the repository first.
   - Use list_files.
   - Read relevant files with read_file.

2. Understand the requirements.

3. Plan briefly.

4. Implement the solution using write_file.

5. Write or update tests where appropriate.

6. Run tests using run_command.

7. If tests fail:
   - inspect the failure
   - identify the cause
   - modify the code
   - rerun the tests
   - continue until the problem is fixed

8. Do NOT ask the pilot for routine actions.

9. Use request_pilot_decision ONLY for meaningful or risky decisions such as:
   - deleting important data
   - security/authentication changes
   - adding dependencies
   - major architectural changes
   - ambiguous decisions that materially change the outcome
   - risky commands

10. If the pilot redirects you, follow the guidance exactly.

11. When the work is complete, call finish.

IMPORTANT:

Never claim that something works unless you verified it.

Never claim tests passed unless you actually ran them.

Give an honest requirement-by-requirement final report.

Human = Pilot.
AI = Executor.

Mission type:
{mission}
"""


# ============================================================
# REQUIREMENTS
# ============================================================

REQUIREMENTS = {

    "BUILD": [
        "Add tasks",
        "Delete tasks",
        "Mark tasks complete",
        "Application runs successfully",
    ],

    "DEBUG": [
        "Reproduce the failure",
        "Identify the root cause",
        "Fix without breaking other behaviour",
        "Existing tests pass",
    ],

    "CHANGE": [
        "Requested change implemented",
        "Existing behaviour preserved",
        "Change covered by tests",
        "Application runs successfully",
    ],
}


DEFAULT_TASK = {

    "BUILD":
        "Build a To-Do application with add, delete and complete-task functionality.",

    "DEBUG":
        "Find and fix the failing behaviour in the repository without introducing regressions.",

    "CHANGE":
        "Modify the existing system to add the requested behaviour while preserving current functionality.",
}


# ============================================================
# GEMINI TOOL DECLARATIONS
# ============================================================

TOOL_DECLARATIONS = [

    types.FunctionDeclaration(
        name="list_files",
        description="List all files in the current engineering workspace.",
        parameters={
            "type": "object",
            "properties": {},
        },
    ),

    types.FunctionDeclaration(
        name="read_file",
        description="Read a text file from the engineering workspace.",
        parameters={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Relative path of the file.",
                }
            },
            "required": ["path"],
        },
    ),

    types.FunctionDeclaration(
        name="write_file",
        description="Create or modify a file in the engineering workspace.",
        parameters={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Relative path of the file.",
                },
                "content": {
                    "type": "string",
                    "description": "Complete file content.",
                },
            },
            "required": ["path", "content"],
        },
    ),

    types.FunctionDeclaration(
        name="run_command",
        description=(
            "Run a command inside the workspace. "
            "Safe commands include python, python3, pytest, ls and cat. "
            "Other commands require pilot approval."
        ),
        parameters={
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "Command to execute.",
                }
            },
            "required": ["command"],
        },
    ),

    types.FunctionDeclaration(
        name="request_pilot_decision",
        description=(
            "Pause execution and ask the human pilot to approve "
            "or redirect a meaningful or risky decision."
        ),
        parameters={
            "type": "object",
            "properties": {
                "summary": {
                    "type": "string",
                    "description": "Explain what decision requires pilot input.",
                }
            },
            "required": ["summary"],
        },
    ),

    types.FunctionDeclaration(
        name="finish",
        description=(
            "Finish the engineering mission and provide an honest "
            "requirement-by-requirement report."
        ),
        parameters={
            "type": "object",
            "properties": {
                "summary": {
                    "type": "string",
                    "description": "Final engineering summary.",
                },
                "requirements": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {
                                "type": "string"
                            },
                            "met": {
                                "type": "boolean"
                            },
                        },
                        "required": [
                            "name",
                            "met"
                        ],
                    },
                },
            },
            "required": [
                "summary",
                "requirements",
            ],
        },
    ),
]


GEMINI_TOOLS = [
    types.Tool(
        function_declarations=TOOL_DECLARATIONS
    )
]


# ============================================================
# MISSION STATE
# ============================================================

class Mission:

    def __init__(self, kind, instruction):

        self.id = uuid.uuid4().hex[:10]

        self.kind = kind
        self.instruction = instruction

        self.dir = WORKSPACES / self.id

        self.dir.mkdir(
            parents=True,
            exist_ok=True
        )

        sample = SAMPLES / kind.lower()

        if kind != "BUILD" and sample.exists():

            shutil.copytree(
                sample,
                self.dir,
                dirs_exist_ok=True
            )

        self.status = "planning"

        self.events = []

        self.checkpoint = None

        self.result = None

        self.files_changed = set()

        self.interventions = 0

        self.iterations = 0

        self.last_test = None

        self.seen_failure = False

        self.lock = threading.Lock()

        self.decision = None

        self.wake = threading.Event()


    def emit(self, text, kind="step"):

        with self.lock:

            self.events.append(
                {
                    "text": text,
                    "kind": kind,
                    "t": int(
                        time.time() * 1000
                    ),
                }
            )


    def set_status(self, status):

        with self.lock:

            self.status = status


    def safe_path(self, relative_path):

        path = (
            self.dir / relative_path
        ).resolve()

        workspace = self.dir.resolve()

        if (
            workspace not in path.parents
            and path != workspace
        ):

            raise ValueError(
                "Path escapes mission workspace"
            )

        return path


    def ask_pilot(self, summary):

        self.wake.clear()

        self.decision = None

        with self.lock:

            self.checkpoint = {
                "body": summary
            }

            self.status = "checkpoint"

        self.emit(
            "Pilot checkpoint reached",
            "pilot"
        )

        answered = self.wake.wait(
            CHECKPOINT_TIMEOUT
        )

        if not answered:

            with self.lock:

                self.checkpoint = None

            self.set_status(
                "executing"
            )

            return (
                "Pilot did not respond in time. "
                "Take the safest and smallest option."
            )

        decision = self.decision

        with self.lock:

            self.checkpoint = None

        self.set_status(
            "executing"
        )

        self.interventions += 1

        if decision["action"] == "redirect":

            self.emit(
                "Pilot redirected the agent",
                "pilot"
            )

            return (
                "PILOT REDIRECTED YOU. "
                "Follow this guidance exactly: "
                + decision["guidance"]
            )

        self.emit(
            "Pilot approved",
            "pilot"
        )

        return (
            "Pilot approved. Continue."
        )


# ============================================================
# TOOL IMPLEMENTATION
# ============================================================

def run_tool(mission, name, args):

    # --------------------------------------------------------
    # LIST FILES
    # --------------------------------------------------------

    if name == "list_files":

        mission.emit(
            "Repository inspected"
        )

        files = []

        for path in sorted(
            mission.dir.rglob("*")
        ):

            if not path.is_file():
                continue

            if "__pycache__" in path.parts:
                continue

            if ".pytest_cache" in path.parts:
                continue

            files.append(
                str(
                    path.relative_to(
                        mission.dir
                    )
                )
            )

        return (
            "\n".join(files)
            if files
            else "(empty workspace)"
        )


    # --------------------------------------------------------
    # READ FILE
    # --------------------------------------------------------

    if name == "read_file":

        path = mission.safe_path(
            args["path"]
        )

        if not path.is_file():

            return (
                f"ERROR: {args['path']} "
                "not found"
            )

        mission.emit(
            f"Reading {args['path']}"
        )

        return path.read_text(
            errors="replace"
        )[:20000]


    # --------------------------------------------------------
    # WRITE FILE
    # --------------------------------------------------------

    if name == "write_file":

        path = mission.safe_path(
            args["path"]
        )

        existed = path.exists()

        path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        path.write_text(
            args["content"],
            encoding="utf-8"
        )

        mission.files_changed.add(
            args["path"]
        )

        if mission.status == "planning":

            mission.set_status(
                "executing"
            )

        if mission.seen_failure:

            mission.emit(
                "Agent adapting",
                "adapt"
            )

        mission.emit(
            (
                "Modifying "
                if existed
                else "Creating "
            )
            + args["path"]
        )

        return (
            f"Wrote {args['path']} "
            f"({len(args['content'])} chars)"
        )


    # --------------------------------------------------------
    # RUN COMMAND
    # --------------------------------------------------------

    if name == "run_command":

        return run_command(
            mission,
            args["command"]
        )


    # --------------------------------------------------------
    # PILOT DECISION
    # --------------------------------------------------------

    if name == "request_pilot_decision":

        return mission.ask_pilot(
            args["summary"]
        )


    return (
        f"ERROR: unknown tool {name}"
    )


# ============================================================
# COMMAND EXECUTION
# ============================================================

def run_command(mission, command):

    try:

        argv = shlex.split(
            command
        )

    except ValueError as e:

        return f"ERROR: {e}"


    if not argv:

        return "ERROR: empty command"


    # --------------------------------------------------------
    # RISKY COMMAND
    # --------------------------------------------------------

    if argv[0] not in SAFE_COMMANDS:

        verdict = mission.ask_pilot(
            "The agent wants to run "
            "a non-allow-listed command:\n"
            + command
        )

        if not verdict.startswith(
            "Pilot approved"
        ):

            return (
                verdict
                + " (command was NOT run)"
            )


    if mission.status == "planning":

        mission.set_status(
            "executing"
        )


    if argv[0] in (
        "python",
        "python3"
    ):

        argv[0] = sys.executable


    is_test = (
        "pytest" in command
        or "unittest" in command
    )


    if is_test:

        if mission.seen_failure:

            mission.emit(
                "Tests rerunning"
            )

        else:

            mission.emit(
                "Tests being executed"
            )

    else:

        mission.emit(
            f"Running: {command[:70]}"
        )


    try:

        result = subprocess.run(
            argv,
            cwd=mission.dir,
            capture_output=True,
            text=True,
            timeout=60,
        )

        output = (
            result.stdout
            + result.stderr
        )[-6000:]

        code = result.returncode


    except subprocess.TimeoutExpired:

        output = (
            "ERROR: command timed out "
            "after 60 seconds"
        )

        code = 124


    except FileNotFoundError:

        output = (
            f"ERROR: command not found: "
            f"{argv[0]}"
        )

        code = 127


    if is_test:

        lines = [
            line
            for line in output.strip().splitlines()
            if line.strip()
        ]

        mission.last_test = {

            "ok":
                code == 0,

            "summary":
                lines[-1]
                if lines
                else "",
        }


    if code != 0:

        mission.seen_failure = True

        mission.emit(
            "Failure detected",
            "failure"
        )

    else:

        if is_test:

            mission.emit(
                "Tests passed",
                "success"
            )


    return (
        f"exit code {code}\n"
        f"{output}"
    )


# ============================================================
# GEMINI REQUEST WITH RETRY
# ============================================================

def generate_with_retry(
    mission,
    contents,
    config
):

    global client

    last_error = None

    for attempt in range(5):

        try:

            mission.emit(
                f"Calling Gemini ({attempt + 1}/5)...",
                "system"
            )

            response = client.models.generate_content(
                model=MODEL,
                contents=contents,
                config=config,
            )

            mission.emit(
                "Gemini responded",
                "success"
            )

            return response

        except Exception as e:

            last_error = e

            print(
                f"\n[GEMINI ERROR - attempt {attempt + 1}]"
            )

            print(
                repr(e)
            )

            print()

            error_text = str(e)

            is_temporary = (
                "503" in error_text
                or "UNAVAILABLE" in error_text
                or "429" in error_text
                or "RESOURCE_EXHAUSTED" in error_text
                or "timeout" in error_text.lower()
                or "timed out" in error_text.lower()
            )

            if not is_temporary:

                raise

            if attempt == 4:

                break

            wait_time = 3 * (
                2 ** attempt
            )

            mission.emit(
                (
                    "Gemini unavailable. "
                    "Retrying "
                    f"({attempt + 2}/5) "
                    f"in {wait_time}s..."
                ),
                "system"
            )

            time.sleep(
                wait_time
            )

    raise last_error


# ============================================================
# BUILD FINAL RESULT
# ============================================================

def build_result(
    mission,
    finished
):

    requirements = (
        finished.get(
            "requirements",
            []
        )
        if finished
        else []
    )

    tests = None

    if mission.last_test:

        if mission.last_test["ok"]:

            tests = (
                "All available tests passed"
            )

        else:

            tests = (
                "Tests failing"
            )

        if mission.last_test["summary"]:

            tests += (
                " ("
                + mission.last_test["summary"]
                + ")"
            )

    elif finished:

        tests = (
            "No tests executed"
        )


    verified = sum(
        1
        for req in requirements
        if req.get("met")
    )

    total = len(
        requirements
    )

    completed = (
        finished is not None
        and (
            not mission.last_test
            or mission.last_test["ok"]
        )
    )

    return {

        "task":
            "Completed"
            if completed
            else "Incomplete",

        "requirements":
            {
                "verified": verified,
                "total": total,
            }
            if requirements
            else None,

        "tests":
            tests,

        "files_modified":
            len(
                mission.files_changed
            ),

        "files":
            sorted(
                mission.files_changed
            ),

        "interventions":
            mission.interventions,

        "iterations":
            mission.iterations,

        "report":
            (
                finished or {}
            ).get("summary"),
    }


# ============================================================
# AGENT LOOP
# ============================================================

def run_agent(mission):

    global client

    try:

        if not client:

            raise RuntimeError(
                "GEMINI_API_KEY is not set."
            )


        mission.emit(
            "Mission received",
            "info"
        )

        mission.emit(
            "Requirements analyzed"
        )


        task = (
            mission.instruction
            or DEFAULT_TASK[
                mission.kind
            ]
        )


        mission.emit(
            "Agent analyzing mission"
        )


        contents = [

            types.Content(
                role="user",
                parts=[
                    types.Part(
                        text=(
                            f"MISSION TYPE: "
                            f"{mission.kind}\n\n"

                            f"DEFAULT ENGINEERING "
                            f"MISSION:\n"
                            f"{DEFAULT_TASK[mission.kind]}\n\n"

                            f"PILOT INSTRUCTION:\n"
                            f"{task}\n\n"

                            "Begin by inspecting the "
                            "workspace."
                        )
                    )
                ],
            )

        ]


        config = types.GenerateContentConfig(

            tools=GEMINI_TOOLS,

            automatic_function_calling=(
                types.AutomaticFunctionCallingConfig(
                    disable=True
                )
            ),
        )


        finished = None


        # ----------------------------------------------------
        # AGENT ITERATIONS
        # ----------------------------------------------------

        for iteration in range(
            MAX_ITERATIONS
        ):

            mission.iterations += 1

            response = generate_with_retry(
                mission,
                contents,
                config
            )


            # ------------------------------------------------
            # Preserve Gemini's COMPLETE response.
            # Important for Gemini 3 thought signatures.
            # ------------------------------------------------

            model_content = (
                response.candidates[
                    0
                ].content
            )

            contents.append(
                model_content
            )


            function_calls = []

            for part in model_content.parts:

                if part.function_call:

                    function_calls.append(
                        part.function_call
                    )


            # ------------------------------------------------
            # NO TOOL CALL
            # ------------------------------------------------

            if not function_calls:

                mission.emit(
                    "Agent produced a response"
                )

                break


            function_response_parts = []


            # ------------------------------------------------
            # EXECUTE TOOL CALLS
            # ------------------------------------------------

            for call in function_calls:

                name = call.name

                args = dict(
                    call.args or {}
                )


                mission.emit(
                    f"Agent using {name}"
                )


                # --------------------------------------------
                # FINISH
                # --------------------------------------------

                if name == "finish":

                    finished = args

                    mission.emit(
                        "Engineering report generated",
                        "success"
                    )

                    function_response_parts.append(
                        types.Part.from_function_response(
                            name=name,
                            response={
                                "result":
                                    "Final report recorded."
                            },
                            id=call.id,
                        )
                    )

                    continue


                # --------------------------------------------
                # NORMAL TOOL
                # --------------------------------------------

                try:

                    output = run_tool(
                        mission,
                        name,
                        args
                    )

                except Exception as e:

                    output = (
                        "ERROR executing tool: "
                        + str(e)
                    )


                function_response_parts.append(
                    types.Part.from_function_response(
                        name=name,
                        response={
                            "result":
                                str(output)
                        },
                        id=call.id,
                    )
                )


            # ------------------------------------------------
            # SEND TOOL RESULTS BACK TO GEMINI
            # ------------------------------------------------

            contents.append(
                types.Content(
                    role="user",
                    parts=function_response_parts
                )
            )


            if finished:

                break


        # ====================================================
        # FINAL RESULT
        # ====================================================

        mission.result = build_result(
            mission,
            finished
        )


        if finished:

            mission.set_status(
                "complete"
            )

            mission.emit(
                "Mission complete",
                "success"
            )

        else:

            mission.result["report"] = (
                mission.result.get(
                    "report"
                )
                or
                "Agent stopped without a final report."
            )

            mission.set_status(
                "failed"
            )

            mission.emit(
                "Agent stopped without a final report",
                "failure"
            )


    except Exception as e:

        import traceback

        traceback.print_exc()


        mission.result = {

            "task":
                "Failed",

            "report":
                f"Backend error: {e}",

            "error":
                str(e),
        }


        mission.emit(
            f"Backend error: {e}",
            "failure"
        )


        mission.set_status(
            "failed"
        )


# ============================================================
# FLASK ROUTES
# ============================================================

@app.get("/")
def index():

    return send_from_directory(
        ROOT,
        "index.html"
    )


@app.get("/api/health")
def health():

    return jsonify(

        ok=True,

        model=MODEL,

        key_set=bool(
            os.environ.get(
                "GEMINI_API_KEY"
            )
        ),
    )


@app.post("/api/mission")
def start_mission():

    body = (
        request.get_json(
            force=True,
            silent=True
        )
        or {}
    )


    kind = str(
        body.get(
            "mission",
            ""
        )
    ).upper()


    if kind not in DEFAULT_TASK:

        return jsonify(
            error=(
                "mission must be "
                "BUILD, DEBUG or CHANGE"
            )
        ), 400


    if not os.environ.get(
        "GEMINI_API_KEY"
    ):

        return jsonify(
            error=(
                "GEMINI_API_KEY is not "
                "set on the server"
            )
        ), 500


    mission = Mission(

        kind,

        str(
            body.get(
                "instruction",
                ""
            )
        ).strip()
    )


    MISSIONS[
        mission.id
    ] = mission


    threading.Thread(
        target=run_agent,
        args=(mission,),
        daemon=True
    ).start()


    return jsonify(

        mission_id=mission.id,

        status=mission.status,

        message="Mission accepted",

        events=[],
    )


@app.get("/api/mission/<mid>")
def poll(mid):

    mission = MISSIONS.get(
        mid
    )


    if not mission:

        return jsonify(
            error="unknown mission"
        ), 404


    after = request.args.get(
        "after",
        0,
        type=int
    )


    with mission.lock:

        return jsonify(

            status=mission.status,

            events=mission.events[
                after:
            ],

            checkpoint=(
                mission.checkpoint
            ),

            result=(
                mission.result
            ),
        )


@app.post("/api/mission/<mid>/decision")
def decide(mid):

    mission = MISSIONS.get(
        mid
    )


    if (
        not mission
        or mission.status
        != "checkpoint"
    ):

        return jsonify(
            error=(
                "no checkpoint "
                "is waiting"
            )
        ), 409


    body = (
        request.get_json(
            force=True,
            silent=True
        )
        or {}
    )


    action = body.get(
        "action"
    )


    if action not in (
        "approve",
        "redirect"
    ):

        return jsonify(
            error=(
                "action must be "
                "approve or redirect"
            )
        ), 400


    guidance = str(
        body.get(
            "guidance",
            ""
        )
    ).strip()


    if (
        action == "redirect"
        and not guidance
    ):

        return jsonify(
            error=(
                "guidance required "
                "for redirect"
            )
        ), 400


    mission.decision = {

        "action":
            action,

        "guidance":
            guidance,
    }


    mission.wake.set()


    return jsonify(
        ok=True
    )


# ============================================================
# START SERVER
# ============================================================

if __name__ == "__main__":

    WORKSPACES.mkdir(
        exist_ok=True
    )

    print()

    print(
        "=" * 60
    )

    print(
        "        AGENTPILOT"
    )

    print(
        "        AI Coding Agent & Engineering Simulator"
    )

    print(
        "=" * 60
    )

    print()

    print(
        f"Model: {MODEL}"
    )

    print(
        "Gemini key:",
        "SET" if api_key else "NOT SET"
    )

    print()

    print(
        "Open: http://127.0.0.1:5000"
    )

    print()

    print(
        "=" * 60
    )

    print()

    app.run(
        debug=False,
        threaded=True,
        port=5000
    )