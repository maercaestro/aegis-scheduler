import React from 'react';
import {
  ProfitComparisonChart,
  ProductionComparisonChart,
  InventoryTrendChart,
  ProductMixChart,
  ProfitTrendChart
} from './Charts';

const ChartsView = ({ datasets }) => {
  return (
    <div className="space-y-6">
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <ProfitComparisonChart datasets={datasets} />
        <ProductionComparisonChart datasets={datasets} />
      </div>
      
      <div className="grid grid-cols-1 gap-6">
        <InventoryTrendChart datasets={datasets} />
      </div>
      
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <ProductMixChart datasets={datasets} />
        <ProfitTrendChart datasets={datasets} />
      </div>
    </div>
  );
};

export default ChartsView;