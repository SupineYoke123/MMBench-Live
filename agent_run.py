import asyncio
from agents import Agent, Runner, SQLiteSession
from agents.model_settings import ModelSettings
from openai import AsyncOpenAI
from agents.mcp import MCPServerSse
from agents import Model, ModelProvider, RunConfig, OpenAIChatCompletionsModel
from project_config import BENCHMARK_SUMMARY_AGENT_MCP_PORT, DATA_ACQUISITION_AGENT_MCP_PORT, QA_GENERATION_AGENT_MCP_PORT, QA_VALIDATE_AGENT_MCP_PORT
from agents.prompts import dynamic_benchmark_prompt, benchmark_summary_prompt, data_acquisition_prompt, qa_generation_and_validate_prompt, qa_generation_prompt, qa_validate_prompt


async def main(query):
    async with MCPServerSse(client_session_timeout_seconds=60*30, params={"url": f"http://localhost:{BENCHMARK_SUMMARY_AGENT_MCP_PORT}/sse", "timeout": 600, "sse_read_timeout": 3600}, cache_tools_list=True) as analyser, \
               MCPServerSse(client_session_timeout_seconds=60*30, params={"url": f"http://localhost:{DATA_ACQUISITION_AGENT_MCP_PORT}/sse", "timeout": 600, "sse_read_timeout": 3600}, cache_tools_list=True) as crawler, \
               MCPServerSse(client_session_timeout_seconds=60*30, params={"url": f"http://localhost:{QA_GENERATION_AGENT_MCP_PORT}/sse", "timeout": 600, "sse_read_timeout": 3600}, cache_tools_list=True) as generator, \
               MCPServerSse(client_session_timeout_seconds=60*30, params={"url": f"http://localhost:{QA_VALIDATE_AGENT_MCP_PORT}/sse", "timeout": 600, "sse_read_timeout": 3600}, cache_tools_list=True) as validater:

        model = OpenAIChatCompletionsModel(
            model="gpt-5-mini",
        )

        config = RunConfig(
            model=model,
            tracing_disabled=True
        )

        session = SQLiteSession("dynamic_benchmark")

        analyser_agent = Agent(
            name = "benchmark summary agent",
            instructions = benchmark_summary_prompt,
            model_settings=ModelSettings(tool_choice="auto"),
            mcp_servers = [analyser]
        )

        analysis_tool = analyser_agent.as_tool(
            tool_name= 'benchmark_summary_agent',
            tool_description= "Parse the user's raw dataset requirements from text, PDF, or other reference files, structure them into a standardized dataset requirement document, and save the result as a JSON file."
        )
        
        crawler_agent = Agent(
            name = "data acquisition agent",
            instructions = data_acquisition_prompt,
            model_settings=ModelSettings(tool_choice="auto"),
            mcp_servers = [crawler]
        )

        crawler_tool = crawler_agent.as_tool(
            tool_name= 'data_acquisition_agent',
            tool_description= 'Generate search queries from structured dataset requirements, plan data collection aspects and quantities, crawl/download relevant images, and save image metadata to a JSON file.'
        )

        generator_agent = Agent(
            name = "qa generation agent",
            instructions = qa_generation_prompt,
            model_settings=ModelSettings(tool_choice="auto"),
            mcp_servers = [generator]
        )

        generator_tool = generator_agent.as_tool(
            tool_name= 'qa_generation_agent',
            tool_description= 'Select and call appropriate visual analysis tools to annotate collected images according to the dataset requirements, save annotation results to config_path, convert them into the final user-required JSON format, save the final result to save_path, and return the annotation workflow.'
        )

        validater_agent = Agent(
            name = "qa validate agent",
            instructions = qa_validate_prompt,
            model_settings=ModelSettings(tool_choice="auto"),
            mcp_servers = [validater]
        )

        validater_tool = validater_agent.as_tool(
            tool_name= 'qa_validate_agent',
            tool_description= 'Select and call appropriate visual analysis tools to annotate collected images according to the dataset requirements, save annotation results to config_path, convert them into the final user-required JSON format, save the final result to save_path, and return the annotation workflow.'
        )

        qa_generator_agent = Agent(
            name = "qa generate and validate agent",
            instructions = qa_generation_and_validate_prompt,
            model_settings=ModelSettings(tool_choice="auto"),
            tools=[generator_tool, validater_tool]
        )

        qa_generator_tool = qa_generator_agent.as_tool(
            tool_name= 'qa_validate_agent',
            tool_description= 'Select and call appropriate visual analysis tools to annotate collected images according to the dataset requirements, save annotation results to config_path, convert them into the final user-required JSON format, save the final result to save_path, and return the annotation workflow.'
        )


        dynamic_benchmark_agent = Agent(
            name = 'dynamic benchmark agent.',
            instructions = dynamic_benchmark_prompt,
            model_settings=ModelSettings(tool_choice="auto"),
            tools=[analysis_tool, crawler_tool, qa_generator_tool]
        )

        result = await Runner.run(dynamic_benchmark_agent, query, run_config=config, session=session)

        return result.final_output