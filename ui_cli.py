from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.prompt import Prompt
from rich.spinner import Spinner
from rich.live import Live

console = Console()


def chat_loop(memory, retriever, prompt, llm):

    console.print(
        Panel.fit(
            "[bold cyan]Stoic Mentor — Marcus AI[/bold cyan]\n"
            "[dim]A calm companion inspired by Meditations[/dim]",
            border_style="cyan"
        )
    )

    console.print(
        "[dim]Commands: /exit  /clear  /help[/dim]\n"
    )

    while True:

        query = Prompt.ask(
            "[bold green]You[/bold green]"
        )

        # =========================
        # COMMANDS
        # =========================

        if query.lower() in ["/exit", "exit", "0"]:
            console.print(
                "\n[bold cyan]Marcus:[/bold cyan] Farewell. Guard your mind well.\n"
            )
            break

        if query.lower() == "/clear":
            memory.clear()

            console.print(
                "[yellow]Conversation memory cleared.[/yellow]\n"
            )

            continue

        if query.lower() == "/help":

            console.print(
                Panel(
                    """
[bold]/exit[/bold]  → Exit chatbot
[bold]/clear[/bold] → Clear conversation memory
[bold]/help[/bold]  → Show commands
                    """,
                    title="Commands",
                    border_style="blue"
                )
            )

            continue

        # =========================
        # LOAD MEMORY
        # =========================

        chat_history = memory.load_memory_variables({})

        enhanced_query = f"""
Conversation:
{chat_history["history"]}

Current question:
{query}
"""

        # =========================
        # RETRIEVE DOCS
        # =========================

        with Live(
            Spinner("dots", text="Marcus is reflecting..."),
            refresh_per_second=10,
            console=console
        ):

            docs = retriever.invoke(enhanced_query)

            context = "\n\n".join(
                [
                    f"[Teaching {i+1}]\n{doc.page_content}"
                    for i, doc in enumerate(docs)
                ]
            )

            # =========================
            # BUILD PROMPT
            # =========================

            final_prompt = prompt.invoke({
                "chat_history": chat_history["history"],
                "context": context,
                "question": query
            })

            # =========================
            # GENERATE RESPONSE
            # =========================

            response = llm.invoke(final_prompt)

        # =========================
        # DISPLAY RESPONSE
        # =========================

        console.print()

        console.print(
            Panel(
                Markdown(response.content),
                title="[bold cyan]Marcus[/bold cyan]",
                border_style="cyan"
            )
        )

        console.print()

        # =========================
        # SAVE MEMORY
        # =========================

        memory.save_context(
            {"input": query},
            {"output": str(response.content)}
        )