"use client";

import { BarChart, LineChart, ScatterChart } from "echarts/charts";
import { GridComponent, LegendComponent, TooltipComponent } from "echarts/components";
import * as echarts from "echarts/core";
import { SVGRenderer } from "echarts/renderers";
import type { EChartsOption } from "echarts";
import EChartsCore from "echarts-for-react/lib/core";

// only the pieces the lab draws, so the chunk carries line and bar charts and nothing else
echarts.use([LineChart, BarChart, ScatterChart, GridComponent, LegendComponent, TooltipComponent, SVGRenderer]);

export type EChartClick = (params: { seriesId?: string; seriesName?: string; dataIndex: number; name?: string }) => void;

/** The ECharts binding itself, loaded on demand by LabEChart in Charts.tsx. */
export default function EChartView({ option, height, onClick }: { option: EChartsOption; height: number; onClick?: EChartClick }) {
  return <EChartsCore echarts={echarts} option={option} style={{ height, width: "100%" }} opts={{ renderer: "svg" }} onEvents={onClick ? { click: onClick } : undefined} />;
}
