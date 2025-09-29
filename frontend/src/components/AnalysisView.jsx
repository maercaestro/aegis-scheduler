import React, { useState } from 'react';
import { TrendingUp, TrendingDown, BarChart3, PieChart, Calendar, Calculator } from 'lucide-react';

const MetricCard = ({ title, value, change, changeType, icon: Icon, className = "" }) => (
  <div className={`bg-white rounded-lg shadow p-6 ${className}`}>
    <div className="flex items-center justify-between">
      <div>
        <p className="text-sm font-medium text-gray-600">{title}</p>
        <p className="text-2xl font-semibold text-gray-900">{value}</p>
        {change !== undefined && (
          <div className={`flex items-center mt-2 text-sm ${
            changeType === 'positive' ? 'text-green-600' : changeType === 'negative' ? 'text-red-600' : 'text-gray-600'
          }`}>
            {changeType === 'positive' && <TrendingUp className="h-4 w-4 mr-1" />}
            {changeType === 'negative' && <TrendingDown className="h-4 w-4 mr-1" />}
            {change}
          </div>
        )}
      </div>
      {Icon && <Icon className="h-8 w-8 text-gray-400" />}
    </div>
  </div>
);

const AnalysisCard = ({ title, children, className = "" }) => (
  <div className={`bg-white rounded-lg shadow p-6 ${className}`}>
    <h3 className="text-lg font-semibold text-gray-800 mb-4">{title}</h3>
    {children}
  </div>
);

const RecommendationCard = ({ title, recommendations, type = "info" }) => {
  const colors = {
    info: 'border-blue-200 bg-blue-50',
    success: 'border-green-200 bg-green-50',
    warning: 'border-yellow-200 bg-yellow-50',
    error: 'border-red-200 bg-red-50'
  };

  return (
    <div className={`border-l-4 p-4 ${colors[type]}`}>
      <h4 className="font-medium text-gray-900 mb-2">{title}</h4>
      <ul className="space-y-1">
        {recommendations.map((rec, index) => (
          <li key={index} className="text-sm text-gray-700 flex items-start">
            <span className="inline-block w-1 h-1 bg-gray-400 rounded-full mt-2 mr-2 flex-shrink-0"></span>
            {rec}
          </li>
        ))}
      </ul>
    </div>
  );
};

const AnalysisView = ({ datasets, summaries }) => {
  const { case1, case2, excel } = datasets;
  const [selectedTimeframe, setSelectedTimeframe] = useState('all');

  // Calculate advanced metrics
  const calculateAdvancedMetrics = () => {
    if (!case1 || !case2) return null;

    const case1Metrics = {
      totalProfit: case1.reduce((sum, row) => sum + (parseFloat(row.Profit) || 0), 0),
      totalProduction: case1.reduce((sum, row) => sum + (parseFloat(row['Quantity Produced']) || 0), 0),
      avgInventory: case1.reduce((sum, row) => sum + (parseFloat(row['Inventory Available']) || 0), 0) / case1.length,
      avgUllage: case1.reduce((sum, row) => sum + (parseFloat(row.Ullage) || 0), 0) / case1.length
    };

    const case2Metrics = {
      totalProfit: case2.reduce((sum, row) => sum + (parseFloat(row.Profit) || 0), 0),
      totalProduction: case2.reduce((sum, row) => sum + (parseFloat(row['Quantity Produced']) || 0), 0),
      avgInventory: case2.reduce((sum, row) => sum + (parseFloat(row['Inventory Available']) || 0), 0) / case2.length,
      avgUllage: case2.reduce((sum, row) => sum + (parseFloat(row.Ullage) || 0), 0) / case2.length
    };

    return {
      profitImprovement: ((case2Metrics.totalProfit - case1Metrics.totalProfit) / case1Metrics.totalProfit) * 100,
      productionImprovement: ((case2Metrics.totalProduction - case1Metrics.totalProduction) / case1Metrics.totalProduction) * 100,
      inventoryEfficiency: ((case1Metrics.avgInventory - case2Metrics.avgInventory) / case1Metrics.avgInventory) * 100,
      case1Metrics,
      case2Metrics
    };
  };

  const metrics = calculateAdvancedMetrics();

  const generateRecommendations = () => {
    if (!metrics) return [];

    const recommendations = [];

    if (metrics.profitImprovement > 5) {
      recommendations.push({
        type: 'success',
        title: 'Case 2 Shows Superior Performance',
        items: [
          `Case 2 generates ${metrics.profitImprovement.toFixed(1)}% higher profit than Case 1`,
          'Consider implementing Case 2 strategy as primary optimization approach',
          'Analyze specific operational changes that drive this improvement'
        ]
      });
    }

    if (metrics.productionImprovement > 2) {
      recommendations.push({
        type: 'info',
        title: 'Production Optimization Opportunities',
        items: [
          `Case 2 achieves ${metrics.productionImprovement.toFixed(1)}% higher production`,
          'Review resource allocation and scheduling efficiency',
          'Consider scaling successful production patterns'
        ]
      });
    }

    if (Math.abs(metrics.inventoryEfficiency) > 5) {
      recommendations.push({
        type: metrics.inventoryEfficiency > 0 ? 'success' : 'warning',
        title: 'Inventory Management Insights',
        items: [
          `Case ${metrics.inventoryEfficiency > 0 ? '2' : '1'} maintains ${Math.abs(metrics.inventoryEfficiency).toFixed(1)}% more efficient inventory levels`,
          'Review inventory holding strategies and costs',
          'Balance between storage capacity and operational flexibility'
        ]
      });
    }

    // Product mix analysis
    if (case1 && case2) {
      const case1Products = [...new Set(case1.map(row => row['Final Product']).filter(Boolean))];
      const case2Products = [...new Set(case2.map(row => row['Final Product']).filter(Boolean))];
      
      if (case1Products.length !== case2Products.length) {
        recommendations.push({
          type: 'info',
          title: 'Product Mix Diversity',
          items: [
            `Case ${case1Products.length > case2Products.length ? '1' : '2'} produces ${Math.abs(case1Products.length - case2Products.length)} more product types`,
            'Evaluate trade-offs between product diversity and operational efficiency',
            'Consider market demand patterns when selecting optimal product mix'
          ]
        });
      }
    }

    return recommendations;
  };

  const recommendations = generateRecommendations();

  if (!case1 && !case2 && !excel) {
    return (
      <div className="bg-white rounded-lg shadow p-6">
        <div className="text-center py-12">
          <Calculator className="h-16 w-16 text-gray-400 mx-auto mb-4" />
          <h3 className="text-lg font-medium text-gray-900 mb-2">Advanced Analysis</h3>
          <p className="text-gray-500">
            Load data files to see detailed analysis, recommendations, and insights.
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Key Metrics Overview */}
      {metrics && (
        <>
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
            <MetricCard
              title="Profit Improvement"
              value={`${metrics.profitImprovement > 0 ? '+' : ''}${metrics.profitImprovement.toFixed(1)}%`}
              change={`$${(metrics.case2Metrics.totalProfit - metrics.case1Metrics.totalProfit).toLocaleString()}`}
              changeType={metrics.profitImprovement > 0 ? 'positive' : 'negative'}
              icon={TrendingUp}
              className="border-l-4 border-green-500"
            />
            <MetricCard
              title="Production Change"
              value={`${metrics.productionImprovement > 0 ? '+' : ''}${metrics.productionImprovement.toFixed(1)}%`}
              change={`${(metrics.case2Metrics.totalProduction - metrics.case1Metrics.totalProduction).toFixed(1)} units`}
              changeType={metrics.productionImprovement > 0 ? 'positive' : 'negative'}
              icon={BarChart3}
              className="border-l-4 border-blue-500"
            />
            <MetricCard
              title="Inventory Efficiency"
              value={`${metrics.inventoryEfficiency > 0 ? '+' : ''}${metrics.inventoryEfficiency.toFixed(1)}%`}
              change={`${metrics.inventoryEfficiency > 0 ? 'More efficient' : 'Higher inventory'}`}
              changeType={metrics.inventoryEfficiency > 0 ? 'positive' : 'negative'}
              icon={PieChart}
              className="border-l-4 border-purple-500"
            />
            <MetricCard
              title="Time Period"
              value={case1 ? `${case1.length} days` : 'N/A'}
              change={excel ? `vs ${excel.length} days (Excel)` : ''}
              icon={Calendar}
              className="border-l-4 border-orange-500"
            />
          </div>

          {/* Detailed Analysis */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <AnalysisCard title="Performance Comparison" className="border-l-4 border-blue-500">
              <div className="space-y-4">
                <div>
                  <h4 className="font-medium text-gray-900 mb-2">Profitability Analysis</h4>
                  <div className="space-y-2 text-sm">
                    <div className="flex justify-between">
                      <span>Case 1 Total Profit:</span>
                      <span className="font-medium">${metrics.case1Metrics.totalProfit.toLocaleString()}</span>
                    </div>
                    <div className="flex justify-between">
                      <span>Case 2 Total Profit:</span>
                      <span className="font-medium">${metrics.case2Metrics.totalProfit.toLocaleString()}</span>
                    </div>
                    <div className="flex justify-between text-lg font-semibold pt-2 border-t">
                      <span>Difference:</span>
                      <span className={metrics.profitImprovement > 0 ? 'text-green-600' : 'text-red-600'}>
                        ${(metrics.case2Metrics.totalProfit - metrics.case1Metrics.totalProfit).toLocaleString()}
                      </span>
                    </div>
                  </div>
                </div>
                
                <div>
                  <h4 className="font-medium text-gray-900 mb-2">Operational Efficiency</h4>
                  <div className="space-y-2 text-sm">
                    <div className="flex justify-between">
                      <span>Case 1 Avg Inventory:</span>
                      <span className="font-medium">{metrics.case1Metrics.avgInventory.toFixed(1)}</span>
                    </div>
                    <div className="flex justify-between">
                      <span>Case 2 Avg Inventory:</span>
                      <span className="font-medium">{metrics.case2Metrics.avgInventory.toFixed(1)}</span>
                    </div>
                    <div className="flex justify-between">
                      <span>Case 1 Avg Ullage:</span>
                      <span className="font-medium">{metrics.case1Metrics.avgUllage.toFixed(1)}</span>
                    </div>
                    <div className="flex justify-between">
                      <span>Case 2 Avg Ullage:</span>
                      <span className="font-medium">{metrics.case2Metrics.avgUllage.toFixed(1)}</span>
                    </div>
                  </div>
                </div>
              </div>
            </AnalysisCard>

            <AnalysisCard title="Strategic Insights" className="border-l-4 border-green-500">
              <div className="space-y-4">
                <div>
                  <h4 className="font-medium text-gray-900 mb-2">Key Findings</h4>
                  <ul className="space-y-2 text-sm text-gray-700">
                    <li className="flex items-start">
                      <span className="inline-block w-2 h-2 bg-blue-500 rounded-full mt-1.5 mr-2 flex-shrink-0"></span>
                      Case 2 {metrics.profitImprovement > 0 ? 'outperforms' : 'underperforms'} Case 1 by {Math.abs(metrics.profitImprovement).toFixed(1)}% in profitability
                    </li>
                    <li className="flex items-start">
                      <span className="inline-block w-2 h-2 bg-green-500 rounded-full mt-1.5 mr-2 flex-shrink-0"></span>
                      Production efficiency shows {Math.abs(metrics.productionImprovement).toFixed(1)}% {metrics.productionImprovement > 0 ? 'improvement' : 'decline'}
                    </li>
                    <li className="flex items-start">
                      <span className="inline-block w-2 h-2 bg-purple-500 rounded-full mt-1.5 mr-2 flex-shrink-0"></span>
                      Inventory management is {Math.abs(metrics.inventoryEfficiency).toFixed(1)}% {metrics.inventoryEfficiency > 0 ? 'more efficient' : 'less efficient'}
                    </li>
                  </ul>
                </div>
                
                <div>
                  <h4 className="font-medium text-gray-900 mb-2">Optimization Potential</h4>
                  <div className="text-sm text-gray-700">
                    <p>
                      The analysis suggests that {metrics.profitImprovement > 0 ? 'Case 2' : 'Case 1'} represents 
                      the more optimal operational strategy, with potential for further improvements in:
                    </p>
                    <ul className="mt-2 space-y-1">
                      <li>• Resource allocation efficiency</li>
                      <li>• Inventory turnover optimization</li>
                      <li>• Product mix refinement</li>
                    </ul>
                  </div>
                </div>
              </div>
            </AnalysisCard>
          </div>
        </>
      )}

      {/* Recommendations */}
      {recommendations.length > 0 && (
        <AnalysisCard title="Strategic Recommendations" className="border-l-4 border-orange-500">
          <div className="space-y-4">
            {recommendations.map((rec, index) => (
              <RecommendationCard
                key={index}
                title={rec.title}
                recommendations={rec.items}
                type={rec.type}
              />
            ))}
          </div>
        </AnalysisCard>
      )}

      {/* Excel Data Analysis (if available) */}
      {excel && (
        <AnalysisCard title="Base Case (Excel) Analysis" className="border-l-4 border-green-500">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="text-center p-4 bg-gray-50 rounded-lg">
              <div className="text-2xl font-bold text-green-600">
                {excel.length}
              </div>
              <div className="text-sm text-gray-600">Days of Data</div>
            </div>
            <div className="text-center p-4 bg-gray-50 rounded-lg">
              <div className="text-2xl font-bold text-blue-600">
                {(excel.reduce((sum, row) => sum + (parseFloat(row['Processing Inventory']) || 0), 0) / excel.length).toFixed(0)}
              </div>
              <div className="text-sm text-gray-600">Avg Processing Inventory</div>
            </div>
            <div className="text-center p-4 bg-gray-50 rounded-lg">
              <div className="text-2xl font-bold text-purple-600">
                {(excel.reduce((sum, row) => sum + (parseFloat(row.Ullage) || 0), 0) / excel.length).toFixed(0)}
              </div>
              <div className="text-sm text-gray-600">Avg Ullage</div>
            </div>
          </div>
          
          <div className="mt-4 text-sm text-gray-700">
            <p>
              The Base Case data provides a reference point for inventory management and processing patterns.
              Compare these operational parameters with the optimized CSV scenarios to identify improvement opportunities.
            </p>
          </div>
        </AnalysisCard>
      )}
    </div>
  );
};

export default AnalysisView;