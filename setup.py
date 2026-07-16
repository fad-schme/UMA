from pathlib import Path
import shutil

from setuptools import find_namespace_packages, setup
from setuptools.command.build_py import build_py as _build_py


ROOT = Path(__file__).parent
README = (ROOT / "README.md").read_text(encoding="utf-8")

version_ns = {}
exec((ROOT / "uma" / "version.py").read_text(encoding="utf-8"), version_ns)


class build_py(_build_py):
    """Clear stale generated package trees before copying current sources."""

    def run(self):
        build_root = Path(self.build_lib) / "uma"
        if build_root.exists():
            shutil.rmtree(build_root)
        super().run()


setup(
    name="uma",
    version=version_ns["__version__"],
    description="UMA: modular memory and context manager for AI agents",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/your-org/uma-rlm",
    author="a.Diaz-Schmeier",
    author_email="ad-schme@ai-mem-engineering.com",
    license="Apache-2.0",
    python_requires=">=3.9",
    packages=find_namespace_packages(include=["uma", "uma.*"]),
    include_package_data=True,
    package_data={"uma.common": ["injection_patterns*.yaml"]},
    install_requires=[
        "pyyaml",
        "numpy",
        "aiohttp",
        "charset-normalizer",
        "pydantic>=2",
        "openai>=1.0.0",
        "lancedb>=0.25.3,<0.26",
    ],
    extras_require={
        "llm": ["anthropic"],
        "vector": ["faiss-cpu", "weaviate-client", "pinecone-client", "fastembed"],
        "graph": ["neo4j"],
        "postgres": ["psycopg2-binary>=2.9"],
        "dev": ["pytest", "matplotlib", "PyPDF2", "pytest-asyncio"],
        "ollama": ["ollama"],
        "parsers": ["PyPDF2", "beautifulsoup4", "markdown", "pandas"],
    },
    cmdclass={"build_py": build_py},
)
