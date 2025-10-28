from typing import Optional, Tuple
from multimeditron.cli import EPILOG, main_cli
import click

def split_host_port(hostport: str, default_port: Optional[int] = None) -> Tuple[str, int]:
    if ':' in hostport:
        host, port_str = hostport.rsplit(':', 1)
        try:
            port = int(port_str)
        except ValueError:
            raise ValueError(f"Invalid port number: {port_str}")
        assert port > 0 and port < 65536, "Port number must be between 1 and 65535"

    else:
        host = hostport
        if default_port is not None:
            port = default_port
        else:
            raise ValueError("Port number is required if not provided in hostport and no default_port is set.")

    return host, port

@main_cli.command("tokenizer_set_chat_template", epilog=EPILOG)
@click.option("--output", "-o", type=click.Path(), required=True, help="Path to save the final tokenizer with overwritten chat template.")
@click.option("--chat-template", "-t", type=click.Path(file_okay=True, exists=True), required=True, help="Path to the chat template file.")
@click.option("--tokenizer-path", "-p", type=str, required=True, help="Path to the tokenizer to modify.")
def set_chat_template(
    output: str,
    chat_template: str,
    tokenizer_path: str,
):
    """
    Load a tokenizer from `tokenizer_path`, overwrite its chat template with the content of `chat_template`,
    and save the modified tokenizer to `output`.
    """
    from transformers import AutoTokenizer
    import logging
    logger = logging.getLogger(__name__)

    logger.info(f"Loading tokenizer from {tokenizer_path}...")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)

    logger.info(f"Reading chat template from {chat_template}...")
    with open(chat_template, "r") as f:
        template_content = f.read()
    tokenizer.chat_template = template_content
    if tokenizer.pad_token is None:
        print("No pad token found, set pad token to eos token")
        tokenizer.pad_token = tokenizer.eos_token

    logger.info(f"Saving modified tokenizer to {output}...")
    tokenizer.save_pretrained(output)
    logger.info(f"Modified tokenizer saved to {output}")
