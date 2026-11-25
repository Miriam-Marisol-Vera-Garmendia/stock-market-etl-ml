"""Agente de dominio Features/ML: auditoría y explicación del trabajo de ML."""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

from claude_agent_sdk import tool

from src.agents_mcp.domain_agent import DomainAgent
from src.agents_mcp.domains.ml_service import MLService
from src.semantic.models import json_safe

ROOT = Path(__file__).resolve().parents[3]

SYSTEM_PROMPT = """Eres el agente de dominio Features/ML. Explica la evidencia real
registrada por el equipo, distingue producción, investigación y descartado, y no
presentes una salida experimental como recomendación de inversión."""

MODEL_IDS = [
    "supermodel", "ridge_rank", "lightgbm_ranking", "woe", "lstm_tcn",
    "regimen_hmm", "ising", "noticias_sentimiento", "reversion_5d",
    "vol_targeting",
]


def _reply(payload: dict) -> dict:
    return {
        "content": [{
            "type": "text",
            "text": json.dumps(json_safe(payload), ensure_ascii=False, allow_nan=False),
        }]
    }


def _tool_error(service: MLService, exc: Exception) -> dict:
    return _reply(service._error("argumento_invalido", f"Error controlado: {exc}"))


def create_ml_tools(root=ROOT, *, service: MLService | None = None):
    """Construye tools aislables y de solo lectura para Features/ML."""
    service = service or MLService(Path(root))

    @tool("schema_features_ml", "Describe el catálogo, estados y reglas del dominio Features/ML.", {})
    async def schema_features_ml(args: dict) -> dict:
        return _reply(await asyncio.to_thread(service.schema))

    @tool(
        "registro_experimentos",
        "Consulta la bitácora de trials, con N global y por familia.",
        {
            "type": "object",
            "properties": {
                "familia": {"type": "string"},
                "prefijo": {"type": "string"},
                "limite": {"type": "integer", "minimum": 1, "maximum": 50, "default": 10},
            },
            "required": [],
            "additionalProperties": False,
        },
    )
    async def registro_experimentos(args: dict) -> dict:
        try:
            result = await asyncio.to_thread(
                service.registro_experimentos,
                args.get("familia"), args.get("prefijo"), args.get("limite", 10),
            )
            return _reply(result)
        except Exception as exc:
            return _tool_error(service, exc)

    @tool(
        "veredicto_modelo",
        "Devuelve estado, evidencia, métricas y límites de un modelo del catálogo ML.",
        {
            "type": "object",
            "properties": {"modelo": {"type": "string", "enum": MODEL_IDS}},
            "required": ["modelo"],
            "additionalProperties": False,
        },
    )
    async def veredicto_modelo(args: dict) -> dict:
        try:
            return _reply(
                await asyncio.to_thread(service.veredicto_modelo, args.get("modelo", ""))
            )
        except Exception as exc:
            return _tool_error(service, exc)

    @tool(
        "senales_recientes",
        "Consulta una señal registrada por fecha o la última disponible hasta un corte.",
        {
            "type": "object",
            "properties": {
                "fecha": {"type": "string", "description": "YYYY-MM-DD; opcional."},
                "corte": {"type": "string", "description": "ISO-8601 con zona horaria; opcional."},
            },
            "required": [],
            "additionalProperties": False,
        },
    )
    async def senales_recientes(args: dict) -> dict:
        try:
            result = await asyncio.to_thread(
                service.senales_recientes, args.get("fecha"), args.get("corte")
            )
            return _reply(result)
        except Exception as exc:
            return _tool_error(service, exc)

    @tool(
        "estado_track_record",
        "Resume cuántas señales registradas ya cumplieron su horizonte de 20 días hábiles.",
        {},
    )
    async def estado_track_record(args: dict) -> dict:
        try:
            return _reply(await asyncio.to_thread(service.estado_track_record))
        except Exception as exc:
            return _tool_error(service, exc)

    @tool(
        "calcular_dsr",
        "Calcula Deflated Sharpe con el N global de trials por defecto.",
        {
            "type": "object",
            "properties": {
                "sharpe_anual": {"type": "number"},
                "anios": {"type": "number", "exclusiveMinimum": 0},
                "n_trials": {"type": "integer", "minimum": 1},
                "skew": {"type": "number", "default": 0},
                "kurtosis": {"type": "number", "exclusiveMinimum": 0, "default": 3},
            },
            "required": ["sharpe_anual", "anios"],
            "additionalProperties": False,
        },
    )
    async def calcular_dsr(args: dict) -> dict:
        try:
            result = await asyncio.to_thread(
                service.calcular_dsr,
                args.get("sharpe_anual"),
                args.get("anios"),
                args.get("n_trials"),
                args.get("skew", 0.0),
                args.get("kurtosis", 3.0),
            )
            return _reply(result)
        except Exception as exc:
            return _tool_error(service, exc)

    @tool(
        "plan_experimento",
        "Genera un preregistro de solo lectura y detecta si la configuración ya fue probada.",
        {
            "type": "object",
            "properties": {
                "hipotesis": {"type": "string"},
                "familia": {"type": "string"},
                "config": {"type": "object"},
                "criterio_exito": {"type": "string"},
            },
            "required": ["hipotesis", "familia", "config", "criterio_exito"],
            "additionalProperties": False,
        },
    )
    async def plan_experimento(args: dict) -> dict:
        try:
            result = await asyncio.to_thread(
                service.plan_experimento,
                args.get("hipotesis", ""),
                args.get("familia", ""),
                args.get("config"),
                args.get("criterio_exito", ""),
            )
            return _reply(result)
        except Exception as exc:
            return _tool_error(service, exc)

    @tool(
        "pronostico_precio",
        "Diagnóstico experimental con los cuatro intentos existentes; no es recomendación operativa.",
        {
            "type": "object",
            "properties": {
                "ticker": {"type": "string"},
                "horizonte_dias": {"type": "integer", "default": 42},
            },
            "required": ["ticker"],
            "additionalProperties": False,
        },
    )
    async def pronostico_precio(args: dict) -> dict:
        try:
            result = await asyncio.to_thread(
                service.pronostico_precio,
                args.get("ticker", ""),
                args.get("horizonte_dias", 42),
            )
            return _reply(result)
        except Exception as exc:
            return _tool_error(service, exc)

    @tool(
        "recomendar_portafolio",
        "Lee la demo histórica de optimización; se devuelve como demo no vigente, no recomendación.",
        {},
    )
    async def recomendar_portafolio(args: dict) -> dict:
        try:
            return _reply(await asyncio.to_thread(service.recomendar_portafolio))
        except Exception as exc:
            return _tool_error(service, exc)

    return [
        schema_features_ml,
        registro_experimentos,
        veredicto_modelo,
        senales_recientes,
        estado_track_record,
        calcular_dsr,
        plan_experimento,
        pronostico_precio,
        recomendar_portafolio,
    ]


def create_ml_agent(root=ROOT) -> DomainAgent:
    return DomainAgent(
        name="features-ml",
        system_prompt=SYSTEM_PROMPT,
        tools=create_ml_tools(root),
    )


_DEFAULT_TOOLS = create_ml_tools(ROOT)
(
    schema_features_ml,
    registro_experimentos,
    veredicto_modelo,
    senales_recientes,
    estado_track_record,
    calcular_dsr,
    plan_experimento,
    pronostico_precio,
    recomendar_portafolio,
) = _DEFAULT_TOOLS

features_ml_agent = create_ml_agent(ROOT)
