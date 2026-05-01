from pathlib import Path

from loguru import logger
import typer
import yaml

from ent_cv.config import MODELS_DIR

app = typer.Typer()

NUCLIO_DIR = Path(__file__).resolve().parent / "nuclio"
FUNCTION_YAML_TEMPLATE = (NUCLIO_DIR / "function.yaml.template").read_text()


def _build_spec_block(names: dict | list) -> str:
    """Build the indented JSON spec block for the nuclio annotations field."""
    if isinstance(names, dict):
        entries = sorted(names.items())
    else:
        entries = enumerate(names)

    items = [{"id": k, "name": v, "type": "rectangle"} for k, v in entries]

    lines = ["["]
    for i, item in enumerate(items):
        comma = "," if i < len(items) - 1 else ""
        lines.append(f'  {{ "id": {item["id"]}, "name": "{item["name"]}", "type": "{item["type"]}" }}{comma}')
    lines.append("]")

    indent = "      "  # 6 spaces — aligns under the YAML block scalar '|' marker
    return "\n".join(indent + line for line in lines)


@app.command()
def main(
    model_dir: Path = typer.Argument(MODELS_DIR / "yolo26x_20260220_152329", help="Model directory containing args.yaml"),
    output: Path = NUCLIO_DIR / "function.yaml",
    namespace: str = "cvat",
    name: str = typer.Option(None, "--name", "-n", help="Override the model name used for func/image/annotation naming"),
):
    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    args_yaml_path = model_dir / "args.yaml"
    if not args_yaml_path.exists():
        raise FileNotFoundError(f"args.yaml not found in {model_dir}")

    logger.info(f"Reading training args from: {args_yaml_path}")
    with open(args_yaml_path) as f:
        train_args = yaml.safe_load(f)

    data_yaml_path = Path(train_args["data"])
    if not data_yaml_path.exists():
        raise FileNotFoundError(f"Data yaml not found: {data_yaml_path}")

    logger.info(f"Reading class names from: {data_yaml_path}")
    with open(data_yaml_path) as f:
        data_config = yaml.safe_load(f)

    names = data_config.get("names")
    if not names:
        raise ValueError(f"No 'names' key found in {data_yaml_path}")

    # Use the model name from args.yaml (e.g. 'yolo26x_20260220_152329'), or the --name override.
    model_name = name or train_args.get("name", model_dir.name)
    # DNS/image-safe slug: lowercase, underscores → hyphens.
    model_slug = model_name.lower().replace("_", "-")

    func_name = f"pt-jaehho-{model_slug}"
    image_name = f"cvat.{model_slug}:latest"
    annotation_name = model_name
    base_model_name = train_args.get("name", model_dir.name)
    description = (
        f"ENT surgical instrument detector - {model_name} ({base_model_name}). "
        if name else
        f"ENT surgical instrument detector - {model_name}. "
    )

    weights_path = model_dir / "weights" / "best.pt"
    if not weights_path.exists():
        raise FileNotFoundError(f"Weights file not found: {weights_path}")

    content = FUNCTION_YAML_TEMPLATE.format(
        func_name=func_name,
        namespace=namespace,
        annotation_name=annotation_name,
        spec_block=_build_spec_block(names),
        description=description,
        image_name=image_name,
        weights_path=weights_path,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(content)

    logger.success(f"Generated: {output}")
    logger.info(f"  Function name : {func_name}")
    logger.info(f"  Display name  : {annotation_name}")
    logger.info(f"  Image         : {image_name}")
    logger.info(f"  Classes       : {len(names)}")


if __name__ == "__main__":
    app()
