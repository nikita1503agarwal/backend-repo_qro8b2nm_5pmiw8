import os
from typing import List, Literal, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

app = FastAPI(title="KalkulusWeb Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def read_root():
    return {"message": "KalkulusWeb Backend aktif"}


@app.get("/api/hello")
def hello():
    return {"message": "Halo dari backend API!"}


@app.get("/test")
def test_database():
    """Test endpoint to check if database is available and accessible"""
    response = {
        "backend": "✅ Running",
        "database": "❌ Not Used",
        "database_url": None,
        "database_name": None,
        "connection_status": "Not Connected",
        "collections": [],
    }

    # Check environment variables (for completeness)
    response["database_url"] = "✅ Set" if os.getenv("DATABASE_URL") else "❌ Not Set"
    response["database_name"] = "✅ Set" if os.getenv("DATABASE_NAME") else "❌ Not Set"
    return response


# ==== Calculation Endpoints (Derivative & Integral) ====
class CalcRequest(BaseModel):
    expression: str = Field(..., description="Fungsi dalam variabel x, misal: x**2 + 3*x - 5 atau sin(x)")
    operation: Literal["derivative", "integral"]


class CalcResponse(BaseModel):
    input: str
    operation: str
    result: str


def _symbolic_compute(expr_str: str, op: str) -> str:
    try:
        # Lazy import sympy to speed initial boot
        import sympy as sp

        x = sp.symbols("x")
        # Safe parsing with limited names
        allowed = {
            "x": x,
            "sin": sp.sin,
            "cos": sp.cos,
            "tan": sp.tan,
            "exp": sp.exp,
            "log": sp.log,
            "ln": sp.log,
            "sqrt": sp.sqrt,
            "pi": sp.pi,
            "E": sp.E,
        }
        expr = sp.sympify(expr_str, locals=allowed)

        if op == "derivative":
            res = sp.diff(expr, x)
        else:
            res = sp.integrate(expr, (x))
        return sp.simplify(res).__str__()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Gagal memproses ekspresi: {str(e)}")


@app.post("/api/calculate", response_model=CalcResponse)
def calculate(req: CalcRequest):
    result = _symbolic_compute(req.expression, req.operation)
    return CalcResponse(input=req.expression, operation=req.operation, result=result)


# ==== Plotting Endpoint ====
class PlotRequest(BaseModel):
    mode: Literal["custom", "preset"]
    expression: Optional[str] = None
    preset: Optional[Literal["parabola", "sin", "line", "sqrt", "exp"]] = None
    x_min: float = -10
    x_max: float = 10
    points: int = 300


class PlotResponse(BaseModel):
    expression: str
    x_min: float
    x_max: float
    points: List[List[float]]


PRESETS = {
    "parabola": "x**2",
    "sin": "sin(x)",
    "line": "x",
    "sqrt": "sqrt(x)",
    "exp": "exp(x)",
}


@app.post("/api/plot", response_model=PlotResponse)
def plot_points(req: PlotRequest):
    if req.mode == "custom":
        if not req.expression:
            raise HTTPException(status_code=400, detail="Ekspresi wajib diisi untuk mode custom")
        expr_str = req.expression
    else:
        if not req.preset or req.preset not in PRESETS:
            raise HTTPException(status_code=400, detail="Preset tidak dikenal")
        expr_str = PRESETS[req.preset]

    try:
        import sympy as sp
        import numpy as np

        x = sp.symbols("x")
        allowed = {
            "x": x,
            "sin": sp.sin,
            "cos": sp.cos,
            "tan": sp.tan,
            "exp": sp.exp,
            "log": sp.log,
            "ln": sp.log,
            "sqrt": sp.sqrt,
            "pi": sp.pi,
            "E": sp.E,
        }
        expr = sp.sympify(expr_str, locals=allowed)
        f = sp.lambdify(x, expr, modules=["numpy", {"ln": np.log}])

        xs = np.linspace(req.x_min, req.x_max, req.points)
        ys = f(xs)

        # Handle complex/invalid results by masking to NaN
        xs = np.array(xs, dtype=float)
        try:
            ys = np.array(ys, dtype=float)
        except Exception:
            ys = np.full_like(xs, np.nan, dtype=float)

        pts = []
        for xv, yv in zip(xs.tolist(), ys.tolist()):
            # avoid infs
            if yv is None:
                continue
            try:
                yfloat = float(yv)
                if not (np.isfinite(yfloat)):
                    continue
                pts.append([float(xv), yfloat])
            except Exception:
                continue

        return PlotResponse(expression=str(expr), x_min=req.x_min, x_max=req.x_max, points=pts)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Gagal membuat titik plot: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
