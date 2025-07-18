"""Command-line interface for Redoc."""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional, Dict, Any

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt, Confirm

from redoc import Redoc
from redoc.exceptions import ConversionError

console = Console()

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Redoc - Universal Document Converter",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  # Convert a document
  redoc convert input.pdf output.html
  
  # Convert with template
  redoc template invoice.json -o invoice.pdf
  
  # Interactive mode
  redoc interactive
  """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Convert command
    convert_parser = subparsers.add_parser('convert', help='Convert between document formats')
    convert_parser.add_argument('input', help='Input file path')
    convert_parser.add_argument('output', help='Output file path')
    convert_parser.add_argument('--from', dest='from_format', help='Source format (auto-detected if not specified)')
    convert_parser.add_argument('--to', dest='to_format', help='Target format (inferred from output file if not specified)')
    convert_parser.add_argument('--ocr', action='store_true', help='Enable OCR processing')
    
    # Template command
    template_parser = subparsers.add_parser('template', help='Generate document from template')
    template_parser.add_argument('template', help='Template file or JSON string')
    template_parser.add_argument('-o', '--output', help='Output file path')
    template_parser.add_argument('--format', help='Output format (inferred from output file if not specified)')
    
    # Interactive command
    subparsers.add_parser('interactive', help='Start interactive mode')
    
    # Version command
    subparsers.add_parser('version', help='Show version information')
    
    return parser.parse_args()

def convert_file(redoc: Redoc, args):
    """Handle file conversion."""
    input_path = Path(args.input)
    output_path = Path(args.output)
    
    if not input_path.exists():
        console.print(f"[red]Error: Input file not found: {input_path}")
        return 1
    
    try:
        if args.ocr:
            console.print(f"[yellow]Performing OCR on {input_path}...")
            result = redoc.ocr(input_path, output_file=output_path)
            console.print(f"[green]✓ OCR completed. Output: {output_path}")
        else:
            from_format = args.from_format or input_path.suffix[1:].lower()
            to_format = args.to_format or output_path.suffix[1:].lower()
            
            console.print(f"[yellow]Converting {input_path} ({from_format.upper()}) to {output_path} ({to_format.upper()})...")
            redoc.convert(
                str(input_path),
                to_format=to_format,
                output_file=str(output_path),
                from_format=from_format
            )
            console.print(f"[green]✓ Conversion completed. Output: {output_path}")
        
        return 0
    except ConversionError as e:
        console.print(f"[red]Error: {str(e)}")
        return 1
    except Exception as e:
        console.print(f"[red]Unexpected error: {str(e)}")
        return 1

def process_template(redoc: Redoc, args):
    """Process document template."""
    try:
        # Check if template is a file or JSON string
        template_path = Path(args.template)
        if template_path.exists():
            with open(template_path, 'r') as f:
                template = json.load(f)
        else:
            # Try to parse as JSON string
            template = json.loads(args.template)
        
        # Determine output format
        output_format = args.format
        output_path = None
        
        if args.output:
            output_path = Path(args.output)
            if not output_format:
                output_format = output_path.suffix[1:].lower()
        
        if not output_format:
            output_format = 'pdf'  # Default to PDF
        
        console.print(f"[yellow]Generating document from template...")
        result = redoc.convert(template, output_format, output_file=output_path)
        
        if output_path:
            console.print(f"[green]✓ Document generated: {output_path}")
        else:
            console.print("\n" + "-" * 80)
            console.print(result)
            console.print("-" * 80 + "\n")
        
        return 0
    except json.JSONDecodeError:
        console.print("[red]Error: Invalid JSON in template")
        return 1
    except Exception as e:
        console.print(f"[red]Error processing template: {str(e)}")
        return 1

def interactive_mode():
    """Start interactive mode."""
    console.print(Panel.fit(
        "[bold blue]Redoc - Interactive Mode\n"
        "Convert documents and process templates interactively\n"
        "Type 'exit' or press Ctrl+C to quit",
        title="Welcome"
    ))
    
    redoc = Redoc()
    
    while True:
        try:
            # Show menu
            console.print("\n[bold]Main Menu:")
            console.print("1. Convert document")
            console.print("2. Generate from template")
            console.print("3. Extract content (OCR)")
            console.print("4. Exit")
            
            choice = Prompt.ask("\nChoose an option", choices=["1", "2", "3", "4", "exit"])
            
            if choice == "1" or choice.lower() == "convert":
                input_file = Prompt.ask("Input file path")
                if input_file.lower() == 'exit':
                    continue
                    
                output_file = Prompt.ask("Output file path")
                if output_file.lower() == 'exit':
                    continue
                
                from_format = Prompt.ask("Source format (press Enter to auto-detect)", default="")
                to_format = Prompt.ask("Target format (press Enter to infer from output file)", default="")
                
                args = argparse.Namespace(
                    input=input_file,
                    output=output_file,
                    from_format=from_format if from_format else None,
                    to_format=to_format if to_format else None,
                    ocr=False
                )
                
                convert_file(redoc, args)
                
            elif choice == "2" or choice.lower() == "template":
                template_input = Prompt.ask("Template file path or JSON string")
                if template_input.lower() == 'exit':
                    continue
                    
                output_file = Prompt.ask("Output file path (press Enter to print to console)", default="")
                output_format = Prompt.ask("Output format (press Enter for PDF)", default="pdf")
                
                args = argparse.Namespace(
                    template=template_input,
                    output=output_file if output_file else None,
                    format=output_format if output_format else None
                )
                
                process_template(redoc, args)
                
            elif choice == "3" or choice.lower() in ["ocr", "extract"]:
                input_file = Prompt.ask("Input file (image or PDF) path")
                if input_file.lower() == 'exit':
                    continue
                    
                output_file = Prompt.ask("Output file path (press Enter to print text)", default="")
                
                if output_file:
                    console.print(f"[yellow]Performing OCR on {input_file}...")
                    result = redoc.ocr(input_file, output_file=output_file)
                    console.print(f"[green]✓ OCR completed. Output: {output_file}")
                else:
                    console.print(f"[yellow]Extracting text from {input_file}...")
                    result = redoc.ocr(input_file)
                    console.print("\n" + "-" * 80)
                    console.print(result["text"])
                    console.print("-" * 80 + "\n")
                
            elif choice == "4" or choice.lower() == "exit":
                console.print("[yellow]Goodbye!")
                break
                
        except KeyboardInterrupt:
            console.print("\n[yellow]Operation cancelled.")
            continue
        except Exception as e:
            console.print(f"[red]Error: {str(e)}")
            continue

def main():
    """Main entry point."""
    try:
        args = parse_args()
        redoc = Redoc()
        
        if args.command == 'convert':
            return convert_file(redoc, args)
        elif args.command == 'template':
            return process_template(redoc, args)
        elif args.command == 'interactive':
            interactive_mode()
            return 0
        elif args.command == 'version':
            from redoc import __version__
            console.print(f"[bold]Redoc[/bold] version [cyan]{__version__}")
            return 0
        else:
            # No command provided, show help
            parse_args().parser.print_help()
            return 0
            
    except KeyboardInterrupt:
        console.print("\n[yellow]Operation cancelled by user.")
        return 1
    except Exception as e:
        console.print(f"[red]Error: {str(e)}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
