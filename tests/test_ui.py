from table_ronde import ui


def test_get_agent_color():
    assert ui.get_agent_color("architect") == "bold blue"
    assert ui.get_agent_color("skeptic") == "bold red"
    assert ui.get_agent_color("enthusiast") == "bold green"
    # Fallback
    fallback_color = ui.get_agent_color("unknown_role", 0)
    assert fallback_color in ui.FALLBACK_COLORS


def test_print_banner():
    # Verify no exception is raised
    ui.print_banner("gemini")
    ui.print_banner("")


def test_print_agents_table():
    personas = [
        {
            "role": "skeptic",
            "title": "The Skeptic",
            "emoji": "😈",
            "model": "gemini-3.6-flash",
            "temperature": 0.6,
        },
        {
            "role": "enthusiast",
            "title": "The Enthusiast",
            "emoji": "🚀",
            "model": "gpt-4o",
            "temperature": 0.8,
        },
    ]
    architect_cfg = {
        "role": "architect",
        "title": "The Architect",
        "emoji": "🏛️",
        "model": "gemini-3.6-flash",
        "temperature": 0.3,
    }
    ui.print_agents_table(personas, architect_cfg)


def test_print_config_summary():
    ui.print_config_summary(
        rounds=2, interactive=True, project_path="/tmp", output_path="plan_v2.md"
    )
    ui.print_config_summary(
        rounds=1, interactive=False, project_path=None, output_path="plan_v2.md"
    )


def test_print_phase_and_round_headers():
    ui.print_phase_header("PHASE 1", "Testing description")
    ui.print_round_header(current=1, total=3)
    ui.print_round_header(current=3, total=3)


def test_print_tool_call():
    ui.print_tool_call("search_web", "query='LangChain agents'")
    ui.print_tool_call("read_file")


def test_agent_stream_panel():
    panel = ui.AgentStreamPanel(title="The Skeptic", emoji="😈", role="skeptic")
    # Test internal panel building
    p_initial = panel._build_panel(done=False)
    assert "Thinking..." in str(p_initial.renderable)

    panel.full_text = "# Title\nArgument content"
    p_content = panel._build_panel(done=False)
    assert p_content.subtitle == "[dim]streaming...[/dim]"

    p_done = panel._build_panel(done=True)
    assert p_done.subtitle == "[dim]✓ done[/dim]"


def test_print_final_summary():
    ui.print_final_summary(
        output_path="plan_v2.md",
        transcript_path="transcript.md",
        session_path="session.json",
        duration_secs=125.4,
    )
    ui.print_final_summary(output_path="plan_v2.md")
