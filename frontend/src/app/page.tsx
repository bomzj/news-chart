"use client";

import { useState } from "react";
import { Timeframe } from "../lib/types";
import Chart from "../components/Chart";
import Toolbar from "../components/Toolbar";

export default function Home() {
  const [ticker, setTicker] = useState("BTCUSDT");
  const [timeframe, setTimeframe] = useState<Timeframe>("1d");
  const [minImpact, setMinImpact] = useState(1);

  return (
    <div className="flex flex-col h-full">
      <Toolbar
        ticker={ticker}
        timeframe={timeframe}
        minImpact={minImpact}
        onTickerChange={setTicker}
        onTimeframeChange={setTimeframe}
        onImpactChange={setMinImpact}
      />
      <div className="flex-1 min-h-0">
        <Chart ticker={ticker} timeframe={timeframe} minImpact={minImpact} />
      </div>
    </div>
  );
}
