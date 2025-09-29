import React from 'react';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  BarElement,
  LineElement,
  PointElement,
  Title,
  Tooltip,
  Legend,
  TimeScale,
} from 'chart.js';
import { Bar, Line } from 'react-chartjs-2';
import 'chartjs-adapter-date-fns';

ChartJS.register(
  CategoryScale,
  LinearScale,
  BarElement,
  LineElement,
  PointElement,
  Title,
  Tooltip,
  Legend,
  TimeScale
);

const ChartContainer = ({ title, children, className = "" }) => (
  <div className={`bg-white rounded-lg shadow p-6 ${className}`}>
    <h3 className="text-lg font-semibold text-gray-800 mb-4">{title}</h3>
    <div className="h-64">
      {children}
    </div>
  </div>
);

export const ProfitComparisonChart = ({ datasets }) => {
  const { case1, case2, excel } = datasets;
  
  if (!case1 || !case2) {
    return (
      <ChartContainer title="Total Profit Comparison">
        <div className="flex items-center justify-center h-full text-gray-500">
          Load Case 1 and Case 2 data to see profit comparison
        </div>
      </ChartContainer>
    );
  }

  const case1Profit = case1.reduce((sum, row) => sum + (parseFloat(row.Profit) || 0), 0);
  const case2Profit = case2.reduce((sum, row) => sum + (parseFloat(row.Profit) || 0), 0);

  const data = {
    labels: ['Case 1', 'Case 2'],
    datasets: [{
      label: 'Total Profit ($)',
      data: [case1Profit, case2Profit],
      backgroundColor: ['#3B82F6', '#EF4444'],
      borderColor: ['#2563EB', '#DC2626'],
      borderWidth: 2,
    }]
  };

  const options = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        display: false,
      },
      title: {
        display: false,
      },
      tooltip: {
        callbacks: {
          label: function(context) {
            return `Total Profit: $${context.parsed.y.toLocaleString()}`;
          }
        }
      }
    },
    scales: {
      y: {
        beginAtZero: true,
        ticks: {
          callback: function(value) {
            return '$' + (value / 1000000).toFixed(1) + 'M';
          }
        }
      }
    }
  };

  return (
    <ChartContainer title="Total Profit Comparison">
      <Bar data={data} options={options} />
    </ChartContainer>
  );
};

export const ProductionComparisonChart = ({ datasets }) => {
  const { case1, case2, excel } = datasets;
  
  if (!case1 || !case2) {
    return (
      <ChartContainer title="Total Production Comparison">
        <div className="flex items-center justify-center h-full text-gray-500">
          Load Case 1 and Case 2 data to see production comparison
        </div>
      </ChartContainer>
    );
  }

  const case1Production = case1.reduce((sum, row) => sum + (parseFloat(row['Quantity Produced']) || 0), 0);
  const case2Production = case2.reduce((sum, row) => sum + (parseFloat(row['Quantity Produced']) || 0), 0);
  const excelProcessing = excel ? excel.reduce((sum, row) => sum + (parseFloat(row['Processing Inventory']) || 0), 0) : 0;

  const data = {
    labels: excel ? ['Case 1', 'Case 2', 'Base Case (Excel)'] : ['Case 1', 'Case 2'],
    datasets: [{
      label: 'Total Production/Processing',
      data: excel ? [case1Production, case2Production, excelProcessing] : [case1Production, case2Production],
      backgroundColor: excel ? ['#3B82F6', '#EF4444', '#10B981'] : ['#3B82F6', '#EF4444'],
      borderColor: excel ? ['#2563EB', '#DC2626', '#059669'] : ['#2563EB', '#DC2626'],
      borderWidth: 2,
    }]
  };

  const options = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        display: false,
      },
      title: {
        display: false,
      },
      tooltip: {
        callbacks: {
          label: function(context) {
            return `Total: ${context.parsed.y.toLocaleString()}`;
          }
        }
      }
    },
    scales: {
      y: {
        beginAtZero: true,
      }
    }
  };

  return (
    <ChartContainer title="Total Production Comparison">
      <Bar data={data} options={options} />
    </ChartContainer>
  );
};

export const InventoryTrendChart = ({ datasets }) => {
  const { case1, case2, excel } = datasets;
  
  if (!case1 && !case2 && !excel) {
    return (
      <ChartContainer title="Inventory Trends Over Time">
        <div className="flex items-center justify-center h-full text-gray-500">
          Load data to see inventory trends
        </div>
      </ChartContainer>
    );
  }

  const datasets_chart = [];
  
  if (case1) {
    datasets_chart.push({
      label: 'Case 1 Inventory',
      data: case1.map(row => ({
        x: row.Date,
        y: parseFloat(row['Inventory Available']) || 0
      })),
      borderColor: '#3B82F6',
      backgroundColor: '#3B82F620',
      tension: 0.4,
    });
  }
  
  if (case2) {
    datasets_chart.push({
      label: 'Case 2 Inventory',
      data: case2.map(row => ({
        x: row.Date,
        y: parseFloat(row['Inventory Available']) || 0
      })),
      borderColor: '#EF4444',
      backgroundColor: '#EF444420',
      tension: 0.4,
    });
  }
  
  if (excel) {
    datasets_chart.push({
      label: 'Base Case Processing Inventory',
      data: excel.map(row => ({
        x: row.Date,
        y: parseFloat(row['Processing Inventory']) || 0
      })),
      borderColor: '#10B981',
      backgroundColor: '#10B98120',
      tension: 0.4,
    });
  }

  const data = {
    datasets: datasets_chart
  };

  const options = {
    responsive: true,
    maintainAspectRatio: false,
    interaction: {
      mode: 'index',
      intersect: false,
    },
    plugins: {
      title: {
        display: false,
      },
      tooltip: {
        callbacks: {
          label: function(context) {
            return `${context.dataset.label}: ${context.parsed.y.toLocaleString()}`;
          }
        }
      }
    },
    scales: {
      x: {
        type: 'time',
        time: {
          displayFormats: {
            day: 'MMM dd',
          }
        }
      },
      y: {
        beginAtZero: true,
        title: {
          display: true,
          text: 'Inventory Level'
        }
      }
    }
  };

  return (
    <ChartContainer title="Inventory Trends Over Time">
      <Line data={data} options={options} />
    </ChartContainer>
  );
};

export const ProductMixChart = ({ datasets }) => {
  const { case1, case2 } = datasets;
  
  if (!case1 && !case2) {
    return (
      <ChartContainer title="Product Mix Comparison">
        <div className="flex items-center justify-center h-full text-gray-500">
          Load CSV data to see product mix comparison
        </div>
      </ChartContainer>
    );
  }

  // Calculate product totals for each case
  const calculateProductMix = (data) => {
    const productTotals = {};
    data.forEach(row => {
      const product = row['Final Product'];
      const quantity = parseFloat(row['Quantity Produced']) || 0;
      if (product) {
        productTotals[product] = (productTotals[product] || 0) + quantity;
      }
    });
    return productTotals;
  };

  const case1Products = case1 ? calculateProductMix(case1) : {};
  const case2Products = case2 ? calculateProductMix(case2) : {};
  
  const allProducts = [...new Set([
    ...Object.keys(case1Products),
    ...Object.keys(case2Products)
  ])].sort();

  const data = {
    labels: allProducts,
    datasets: [
      ...(case1 ? [{
        label: 'Case 1',
        data: allProducts.map(product => case1Products[product] || 0),
        backgroundColor: '#3B82F6',
        borderColor: '#2563EB',
        borderWidth: 2,
      }] : []),
      ...(case2 ? [{
        label: 'Case 2',
        data: allProducts.map(product => case2Products[product] || 0),
        backgroundColor: '#EF4444',
        borderColor: '#DC2626',
        borderWidth: 2,
      }] : [])
    ]
  };

  const options = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      title: {
        display: false,
      },
      tooltip: {
        callbacks: {
          label: function(context) {
            return `${context.dataset.label}: ${context.parsed.y.toLocaleString()}`;
          }
        }
      }
    },
    scales: {
      y: {
        beginAtZero: true,
        title: {
          display: true,
          text: 'Total Quantity Produced'
        }
      },
      x: {
        title: {
          display: true,
          text: 'Product Type'
        }
      }
    }
  };

  return (
    <ChartContainer title="Product Mix Comparison">
      <Bar data={data} options={options} />
    </ChartContainer>
  );
};

export const ProfitTrendChart = ({ datasets }) => {
  const { case1, case2 } = datasets;
  
  if (!case1 && !case2) {
    return (
      <ChartContainer title="Daily Profit Trends">
        <div className="flex items-center justify-center h-full text-gray-500">
          Load CSV data to see profit trends
        </div>
      </ChartContainer>
    );
  }

  const datasets_chart = [];
  
  if (case1) {
    datasets_chart.push({
      label: 'Case 1 Daily Profit',
      data: case1.map(row => ({
        x: row.Date,
        y: parseFloat(row.Profit) || 0
      })),
      borderColor: '#3B82F6',
      backgroundColor: '#3B82F620',
      tension: 0.4,
    });
  }
  
  if (case2) {
    datasets_chart.push({
      label: 'Case 2 Daily Profit',
      data: case2.map(row => ({
        x: row.Date,
        y: parseFloat(row.Profit) || 0
      })),
      borderColor: '#EF4444',
      backgroundColor: '#EF444420',
      tension: 0.4,
    });
  }

  const data = {
    datasets: datasets_chart
  };

  const options = {
    responsive: true,
    maintainAspectRatio: false,
    interaction: {
      mode: 'index',
      intersect: false,
    },
    plugins: {
      title: {
        display: false,
      },
      tooltip: {
        callbacks: {
          label: function(context) {
            return `${context.dataset.label}: $${context.parsed.y.toLocaleString()}`;
          }
        }
      }
    },
    scales: {
      x: {
        type: 'time',
        time: {
          displayFormats: {
            day: 'MMM dd',
          }
        }
      },
      y: {
        beginAtZero: true,
        title: {
          display: true,
          text: 'Daily Profit ($)'
        },
        ticks: {
          callback: function(value) {
            return '$' + (value / 1000000).toFixed(1) + 'M';
          }
        }
      }
    }
  };

  return (
    <ChartContainer title="Daily Profit Trends">
      <Line data={data} options={options} />
    </ChartContainer>
  );
};