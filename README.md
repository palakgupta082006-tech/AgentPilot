# AgentPilot

## AI Coding Agent & Engineering Simulator

> **Your AI can code. Can you lead it?**

AgentPilot is an AI coding-agent simulator designed around a simple idea:

**Human = Pilot | AI = Executor**

The developer provides an engineering mission, while the AI agent is designed to explore a repository, plan the work, modify files, execute development tools, test its changes, and report the engineering outcome.

AgentPilot focuses on three engineering workflows:

- **BUILD** — implement a requested feature
- **DEBUG** — investigate and fix a failing application
- **CHANGE** — modify existing behaviour while preserving functionality

---

## Problem

AI coding agents can increasingly perform multi-step engineering tasks, but developers also need to learn how to delegate, supervise, and redirect autonomous agents.

AgentPilot provides a workspace for practicing this interaction.

---

## Solution

AgentPilot places the developer in the role of a **Pilot** while the AI acts as the **Executor**.

The developer defines the goal and remains responsible for the final outcome. The agent is designed to explore the workspace, plan the task, modify code, execute tools, run tests, and iterate.

### Core Workflow

**Set Goal → Delegate → Agent Executes → Supervise → Redirect/Approve → Deliver**

---

## Engineering Missions

### BUILD

The developer gives the agent a new feature or functionality to implement.

Example:

> Add task management functionality with add, delete, and complete operations.

The agent explores the existing project, implements the requested functionality, and verifies the changes.

### DEBUG

The developer provides an existing failure or broken behaviour.

The agent is designed to:

1. Reproduce the failure
2. Investigate the relevant code
3. Identify the root cause
4. Implement a fix
5. Run tests again
6. Verify the result

### CHANGE

The developer requests a modification to existing functionality.

The agent is designed to:

1. Understand the existing implementation
2. Identify affected files
3. Plan the modification
4. Modify the code
5. Update or add tests
6. Verify that existing behaviour is preserved

---

## Key Features

- AI-powered coding agent
- Engineering mission-based workflow
- Repository exploration
- File reading and modification
- Terminal command execution
- Automated testing workflow
- BUILD / DEBUG / CHANGE missions
- Human intervention checkpoints
- Mission event and interaction trace
- Engineering outcome reporting

---

## Technologies Used

- **Python**
- **Flask**
- **Google Gemini API**
- **Gemini Function/Tool Calling**
- **HTML**
- **CSS**
- **JavaScript**
- **Git**
- **GitHub**

---

## Project Structure

```text
AgentPilot/
│
├── app.py
├── agent.py
├── evaluator.py
├── missions.py
├── tools.py
├── index.html
├── requirements.txt
├── README.md
├── .gitignore
├── workspaces/
│
└── .env