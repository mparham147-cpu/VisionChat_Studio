"""
utils/chat_agent.py
===================

Build a LangChain agent with a small memory strategy and invoke it
once per user message.

Three strategies:

* "Recent turns" — keep the last N question/answer pairs.
* "Full history" — send everything.
* "Summarize"    — condense old messages when a token limit is reached.

The agent is rebuilt on every call from the saved history. This is
cheap and avoids duplicating messages when Streamlit reruns the script.
"""

from langchain.agents import create_agent
from langchain.agents.middleware import before_model, SummarizationMiddleware
from langchain.messages import RemoveMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.message import REMOVE_ALL_MESSAGES


def build_agent(chat_model, system_prompt, memory_mode="Recent turns",
                recent_turns=10, summary_trigger_tokens=4000):
    """
    Create a LangChain agent that keeps a chosen memory strategy.

    Parameters
    ----------
    chat_model : LangChain chat model
    system_prompt : str
    memory_mode : "Recent turns" | "Full history" | "Summarize"
    recent_turns : int
    summary_trigger_tokens : int

    Returns
    -------
    A compiled LangChain agent.
    """
    middleware = []

    if memory_mode == "Recent turns":
        @before_model
        def keep_only_recent_messages(state, runtime):
            """Drop everything except the last 2*recent_turns + 1 messages."""
            messages = state["messages"]
            keep_count = 2 * recent_turns + 1
            if len(messages) <= keep_count:
                return None
            return {
                "messages": [
                    RemoveMessage(id=REMOVE_ALL_MESSAGES),
                    *messages[-keep_count:],
                ]
            }
        middleware.append(keep_only_recent_messages)

    elif memory_mode == "Summarize":
        middleware.append(
            SummarizationMiddleware(
                model=chat_model,
                trigger=("tokens", summary_trigger_tokens),
                keep=("messages", max(2, 2 * recent_turns)),
            )
        )

    return create_agent(
        model=chat_model,
        tools=[],
        system_prompt=system_prompt,
        middleware=middleware,
        checkpointer=InMemorySaver(),
    )


def send_message(chat_model, system_prompt, thread_id, history, new_message,
                 memory_mode="Recent turns", recent_turns=10,
                 summary_trigger_tokens=4000):
    """
    Rebuild the agent from history, append the new message, and invoke.

    Returns the final ``AIMessage`` produced by the model.
    """
    agent = build_agent(
        chat_model=chat_model,
        system_prompt=system_prompt,
        memory_mode=memory_mode,
        recent_turns=recent_turns,
        summary_trigger_tokens=summary_trigger_tokens,
    )

    messages = [
        {"role": message["role"],
         "content": message.get("model_content", message["content"])}
        for message in history
    ]
    messages.append({"role": "user", "content": new_message})

    config = {"configurable": {"thread_id": thread_id}}
    result = agent.invoke({"messages": messages}, config=config)
    return result["messages"][-1]
