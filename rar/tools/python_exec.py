"""
PythonExecTool - Safe Python code execution for computation.
"""

import os
import sys
import time
import tempfile
import hashlib
import base64
from io import StringIO
from pathlib import Path
from typing import Optional, Any
from contextlib import redirect_stdout, redirect_stderr

from .base import BaseTool, ToolResult


class PythonExecTool(BaseTool):
    """
    Execute Python code in a sandboxed environment.
    Designed for scientific computation tasks like:
    - Data analysis
    - Curve fitting
    - Statistical calculations
    - Plotting/visualization
    """

    name = "python_exec"
    description = "Execute Python code for computation and analysis"

    # Allowed imports for safety
    ALLOWED_IMPORTS = {
        "math", "statistics", "random", "itertools", "collections",
        "numpy", "np",
        "pandas", "pd",
        "scipy",
        "matplotlib", "matplotlib.pyplot", "plt",
        "json", "csv", "re",
    }

    def __init__(
        self,
        seed: Optional[int] = None,
        timeout: int = 30,
        output_dir: Optional[str] = None
    ):
        super().__init__(seed=seed)
        self.timeout = timeout
        self.output_dir = Path(output_dir) if output_dir else Path(tempfile.gettempdir())

    def _create_safe_globals(self) -> dict:
        """Create a restricted global namespace for execution."""
        safe_globals = {
            "__builtins__": {
                # Safe builtins only
                "abs": abs, "all": all, "any": any, "bool": bool,
                "dict": dict, "enumerate": enumerate, "filter": filter,
                "float": float, "format": format, "frozenset": frozenset,
                "int": int, "isinstance": isinstance, "len": len,
                "list": list, "map": map, "max": max, "min": min,
                "pow": pow, "print": print, "range": range, "reversed": reversed,
                "round": round, "set": set, "sorted": sorted, "str": str,
                "sum": sum, "tuple": tuple, "type": type, "zip": zip,
                "True": True, "False": False, "None": None,
            }
        }

        # Add safe math functions
        import math
        safe_globals["math"] = math

        # Add numpy if available
        try:
            import numpy as np
            safe_globals["numpy"] = np
            safe_globals["np"] = np
            # Set seed for reproducibility
            if self.seed is not None:
                np.random.seed(self.seed)
        except ImportError:
            pass

        # Add pandas if available
        try:
            import pandas as pd
            safe_globals["pandas"] = pd
            safe_globals["pd"] = pd
        except ImportError:
            pass

        # Add scipy if available
        try:
            import scipy
            from scipy import optimize, stats, interpolate
            safe_globals["scipy"] = scipy
            safe_globals["optimize"] = optimize
            safe_globals["stats"] = stats
            safe_globals["interpolate"] = interpolate
        except ImportError:
            pass

        # Add matplotlib if available
        try:
            import matplotlib
            matplotlib.use('Agg')  # Non-interactive backend
            import matplotlib.pyplot as plt
            safe_globals["matplotlib"] = matplotlib
            safe_globals["plt"] = plt
        except ImportError:
            pass

        # Add random with seed
        import random
        if self.seed is not None:
            random.seed(self.seed)
        safe_globals["random"] = random

        # Add statistics
        import statistics
        safe_globals["statistics"] = statistics

        return safe_globals

    def _preprocess_code(self, code: str) -> str:
        """
        Preprocess code to remove import statements for pre-loaded modules.
        Since we pre-load numpy, scipy, etc. in safe_globals, we need to
        strip those import statements.
        """
        import re
        lines = code.split('\n')
        processed_lines = []

        # Patterns for imports we've already provided
        skip_patterns = [
            r'^\s*import\s+numpy(\s+as\s+np)?\s*$',
            r'^\s*import\s+pandas(\s+as\s+pd)?\s*$',
            r'^\s*import\s+scipy\s*$',
            r'^\s*from\s+scipy\s+import\s+',
            r'^\s*import\s+matplotlib',
            r'^\s*from\s+matplotlib',
            r'^\s*import\s+math\s*$',
            r'^\s*import\s+random\s*$',
            r'^\s*import\s+statistics\s*$',
            r'^\s*import\s+np\s*$',
            r'^\s*import\s+pd\s*$',
        ]

        for line in lines:
            skip = False
            for pattern in skip_patterns:
                if re.match(pattern, line):
                    skip = True
                    break
            if not skip:
                processed_lines.append(line)

        return '\n'.join(processed_lines)

    def execute(
        self,
        code: str,
        description: str = "",
        save_figures: bool = True
    ) -> ToolResult:
        """
        Execute Python code safely.

        Args:
            code: Python code to execute
            description: Human-readable description of what the code does
            save_figures: Whether to save matplotlib figures

        Returns:
            ToolResult with execution output and any generated artifacts
        """
        start_time = time.time()
        artifacts = {}

        # Preprocess code to remove redundant imports
        code = self._preprocess_code(code)

        # Capture stdout/stderr
        stdout_capture = StringIO()
        stderr_capture = StringIO()

        # Create safe execution environment
        safe_globals = self._create_safe_globals()
        local_vars = {}

        try:
            # Execute code with captured output
            with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
                exec(code, safe_globals, local_vars)

            stdout_output = stdout_capture.getvalue()
            stderr_output = stderr_capture.getvalue()

            # Check for matplotlib figures
            if save_figures:
                try:
                    import matplotlib.pyplot as plt
                    figures = [plt.figure(i) for i in plt.get_fignums()]
                    for i, fig in enumerate(figures):
                        # Save figure to file
                        fig_path = self.output_dir / f"figure_{i}_{int(time.time())}.png"
                        fig.savefig(fig_path, dpi=100, bbox_inches='tight')

                        # Also encode as base64 for embedding
                        import io
                        buf = io.BytesIO()
                        fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
                        buf.seek(0)
                        img_base64 = base64.b64encode(buf.read()).decode('utf-8')

                        artifacts[f"figure_{i}"] = {
                            "path": str(fig_path),
                            "base64": img_base64,
                            "type": "image/png"
                        }

                    plt.close('all')
                except Exception:
                    pass

            # Extract result variables
            result_vars = {}
            for name, value in local_vars.items():
                if not name.startswith('_'):
                    try:
                        # Try to serialize
                        if isinstance(value, (int, float, str, bool, list, dict)):
                            result_vars[name] = value
                        else:
                            result_vars[name] = str(value)[:500]
                    except Exception:
                        result_vars[name] = "<non-serializable>"

            output = {
                "stdout": stdout_output,
                "stderr": stderr_output,
                "variables": result_vars,
                "success": True
            }

            latency_ms = (time.time() - start_time) * 1000

            return ToolResult(
                success=True,
                output=output,
                tool_name=self.name,
                input_data={"code": code, "description": description},
                latency_ms=latency_ms,
                artifacts=artifacts
            )

        except Exception as e:
            stderr_output = stderr_capture.getvalue()
            latency_ms = (time.time() - start_time) * 1000

            return ToolResult(
                success=False,
                output={
                    "stdout": stdout_capture.getvalue(),
                    "stderr": stderr_output,
                    "error": str(e),
                    "error_type": type(e).__name__
                },
                error=str(e),
                tool_name=self.name,
                input_data={"code": code, "description": description},
                latency_ms=latency_ms
            )

    def get_schema(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "code": {"type": "string", "description": "Python code to execute"},
                "description": {"type": "string", "description": "Description of the computation"},
                "save_figures": {"type": "boolean", "description": "Save matplotlib figures", "default": True}
            }
        }


# Convenience function for quick calculations
def quick_calc(expression: str) -> Any:
    """Evaluate a simple mathematical expression."""
    tool = PythonExecTool()
    result = tool.execute(f"result = {expression}\nprint(result)")
    if result.success:
        return result.output.get("variables", {}).get("result")
    return None
