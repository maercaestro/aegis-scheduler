import React, { useState } from 'react';
import { Upload, FileText, BarChart3, Table, Filter, Download } from 'lucide-react';
import { loadCSV, loadExcel, calculateSummary, compareDatasets } from '../utils/dataLoader';
import ChartsView from './ChartsView';
import TablesView from './TablesView';
import AnalysisView from './AnalysisView';

const FileUploader = ({ onFileLoad, fileType, label, accept }) => {
  const [loading, setLoading] = useState(false);
  const [fileName, setFileName] = useState('');

  const handleFileChange = async (event) => {
    const file = event.target.files[0];
    if (!file) return;

    setLoading(true);
    setFileName(file.name);

    try {
      let data;
      if (fileType === 'csv') {
        data = await loadCSV(file);
      } else if (fileType === 'excel') {
        data = await loadExcel(file);
      }
      
      onFileLoad(data, fileType, file.name);
    } catch (error) {
      console.error('Error loading file:', error);
      alert(`Error loading file: ${error.message}`);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="border-2 border-dashed border-gray-300 hover:border-gray-400 rounded-lg p-6 text-center transition-colors">
      <input
        type="file"
        accept={accept}
        onChange={handleFileChange}
        className="hidden"
        id={`file-${fileType}`}
      />
      <label htmlFor={`file-${fileType}`} className="cursor-pointer">
        <div className="flex flex-col items-center">
          <Upload className={`h-8 w-8 mb-2 ${loading ? 'animate-spin' : ''} text-gray-500`} />
          <p className="text-sm font-medium text-gray-700 mb-1">{label}</p>
          {fileName ? (
            <p className="text-xs text-green-600 flex items-center">
              <FileText className="h-4 w-4 mr-1" />
              {fileName}
            </p>
          ) : (
            <p className="text-xs text-gray-500">Click to upload {accept}</p>
          )}
          {loading && <p className="text-xs text-blue-500 mt-1">Loading...</p>}
        </div>
      </label>
    </div>
  );
};

const SummaryCard = ({ title, data, type, className = "" }) => {
  if (!data) {
    return (
      <div className={`bg-white rounded-lg shadow p-6 ${className}`}>
        <h3 className="text-lg font-semibold text-gray-800 mb-4">{title}</h3>
        <p className="text-gray-500">No data loaded</p>
      </div>
    );
  }

  return (
    <div className={`bg-white rounded-lg shadow p-6 ${className}`}>
      <h3 className="text-lg font-semibold text-gray-800 mb-4">{title}</h3>
      <div className="space-y-3">
        <div className="flex justify-between">
          <span className="text-gray-600">Total Rows:</span>
          <span className="font-medium">{data.totalRows?.toLocaleString()}</span>
        </div>

        {type === 'csv' && (
          <>
            <div className="flex justify-between">
              <span className="text-gray-600">Total Profit:</span>
              <span className="font-medium text-green-600">
                ${data.totalProfit?.toLocaleString(undefined, { maximumFractionDigits: 0 })}
              </span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-600">Total Production:</span>
              <span className="font-medium">{data.totalProduction?.toLocaleString(undefined, { maximumFractionDigits: 1 })}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-600">Avg Profit/Day:</span>
              <span className="font-medium">${data.avgProfit?.toLocaleString(undefined, { maximumFractionDigits: 0 })}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-600">Products:</span>
              <span className="font-medium">{data.products?.length} types</span>
            </div>
            {data.products && (
              <div className="text-xs text-gray-500 mt-2">
                {data.products.join(', ')}
              </div>
            )}
            {data.dateRange && (
              <div className="text-xs text-gray-500 mt-2 pt-2 border-t">
                {data.dateRange.start} to {data.dateRange.end}
              </div>
            )}
          </>
        )}

        {type === 'excel' && (
          <>
            <div className="flex justify-between">
              <span className="text-gray-600">Day Range:</span>
              <span className="font-medium">
                {data.dayRange?.start} - {data.dayRange?.end}
              </span>
            </div>
            {data.avgProcessingInventory && (
              <div className="flex justify-between">
                <span className="text-gray-600">Avg Processing Inv:</span>
                <span className="font-medium">{data.avgProcessingInventory.toLocaleString(undefined, { maximumFractionDigits: 1 })}</span>
              </div>
            )}
            {data.avgUllage && (
              <div className="flex justify-between">
                <span className="text-gray-600">Avg Ullage:</span>
                <span className="font-medium">{data.avgUllage.toLocaleString(undefined, { maximumFractionDigits: 1 })}</span>
              </div>
            )}
            {data.inventoryBreakdown && Object.keys(data.inventoryBreakdown).length > 0 && (
              <div className="mt-4 pt-3 border-t">
                <h4 className="text-sm font-medium text-gray-700 mb-2">Inventory Breakdown:</h4>
                {Object.entries(data.inventoryBreakdown).map(([key, value]) => (
                  <div key={key} className="flex justify-between text-sm">
                    <span className="text-gray-600">{key.replace('Inventory_', '')}:</span>
                    <span>{value.total.toLocaleString(undefined, { maximumFractionDigits: 0 })}</span>
                  </div>
                ))}
              </div>
            )}
          </>
        )}
      </div>
    </div>
  );
};

const DataComparison = () => {
  const [datasets, setDatasets] = useState({
    case1: null,
    case2: null,
    excel: null
  });
  const [summaries, setSummaries] = useState({
    case1: null,
    case2: null,
    excel: null
  });
  const [activeView, setActiveView] = useState('overview');

  const handleFileLoad = (data, fileType, fileName) => {
    const summary = calculateSummary(data, fileType === 'excel' ? 'excel' : 'csv');
    
    if (fileName.toLowerCase().includes('case1')) {
      setDatasets(prev => ({ ...prev, case1: data }));
      setSummaries(prev => ({ ...prev, case1: summary }));
    } else if (fileName.toLowerCase().includes('case2')) {
      setDatasets(prev => ({ ...prev, case2: data }));
      setSummaries(prev => ({ ...prev, case2: summary }));
    } else {
      setDatasets(prev => ({ ...prev, excel: data }));
      setSummaries(prev => ({ ...prev, excel: summary }));
    }
  };

  const comparison = compareDatasets(datasets.case1, datasets.case2, datasets.excel);

  const viewButtons = [
    { id: 'overview', label: 'Overview', icon: BarChart3 },
    { id: 'charts', label: 'Charts', icon: BarChart3 },
    { id: 'tables', label: 'Data Tables', icon: Table },
    { id: 'analysis', label: 'Analysis', icon: Filter },
  ];

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <div className="bg-white shadow-sm border-b">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="py-4">
            <h1 className="text-2xl font-bold text-gray-900">Refinery Data Comparison Dashboard</h1>
            <p className="text-gray-600 mt-1">Compare Case 1, Case 2, and Base Case (Excel) optimization results</p>
          </div>
        </div>
      </div>

      {/* Navigation */}
      <div className="bg-white shadow-sm">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <nav className="flex space-x-8 py-4">
            {viewButtons.map(({ id, label, icon: Icon }) => (
              <button
                key={id}
                onClick={() => setActiveView(id)}
                className={`flex items-center px-3 py-2 rounded-md text-sm font-medium transition-colors ${
                  activeView === id
                    ? 'bg-blue-100 text-blue-700'
                    : 'text-gray-500 hover:text-gray-700 hover:bg-gray-100'
                }`}
              >
                <Icon className="h-4 w-4 mr-2" />
                {label}
              </button>
            ))}
          </nav>
        </div>
      </div>

      {/* Main Content */}
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {activeView === 'overview' && (
          <div className="space-y-8">
            {/* File Upload Section */}
            <div className="bg-white rounded-lg shadow p-6">
              <h2 className="text-xl font-semibold text-gray-800 mb-6">Load Data Files</h2>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                <FileUploader
                  fileType="csv"
                  label="Case 1 CSV"
                  accept=".csv"
                  onFileLoad={handleFileLoad}
                />
                <FileUploader
                  fileType="csv"
                  label="Case 2 CSV"
                  accept=".csv"
                  onFileLoad={handleFileLoad}
                />
                <FileUploader
                  fileType="excel"
                  label="Base Case Excel"
                  accept=".xlsx,.xls"
                  onFileLoad={handleFileLoad}
                />
              </div>
            </div>

            {/* Summary Cards */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
              <SummaryCard
                title="Case 1 Summary"
                data={summaries.case1}
                type="csv"
                className="border-l-4 border-blue-500"
              />
              <SummaryCard
                title="Case 2 Summary"
                data={summaries.case2}
                type="csv"
                className="border-l-4 border-red-500"
              />
              <SummaryCard
                title="Base Case Summary"
                data={summaries.excel}
                type="excel"
                className="border-l-4 border-green-500"
              />
            </div>

            {/* Quick Comparison */}
            {summaries.case1 && summaries.case2 && (
              <div className="bg-white rounded-lg shadow p-6">
                <h2 className="text-xl font-semibold text-gray-800 mb-6">Quick Comparison: Case 1 vs Case 2</h2>
                <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                  <div className="text-center p-4 bg-gray-50 rounded-lg">
                    <div className="text-2xl font-bold text-blue-600">
                      {comparison.comparison?.csv1VsCsv2?.profitPctDiff?.toFixed(1)}%
                    </div>
                    <div className="text-sm text-gray-600 mt-1">Profit Difference</div>
                    <div className="text-xs text-gray-500 mt-1">
                      ${comparison.comparison?.csv1VsCsv2?.profitDiff?.toLocaleString()}
                    </div>
                  </div>
                  <div className="text-center p-4 bg-gray-50 rounded-lg">
                    <div className="text-2xl font-bold text-green-600">
                      {comparison.comparison?.csv1VsCsv2?.productionDiff?.toFixed(1)}
                    </div>
                    <div className="text-sm text-gray-600 mt-1">Production Difference</div>
                  </div>
                  <div className="text-center p-4 bg-gray-50 rounded-lg">
                    <div className="text-2xl font-bold text-purple-600">
                      {summaries.case2.totalProfit > summaries.case1.totalProfit ? 'Case 2' : 'Case 1'}
                    </div>
                    <div className="text-sm text-gray-600 mt-1">Better Performance</div>
                  </div>
                </div>
              </div>
            )}
          </div>
        )}

        {activeView !== 'overview' && (
          <>
            {activeView === 'charts' && <ChartsView datasets={datasets} />}
            {activeView === 'tables' && <TablesView datasets={datasets} />}
            {activeView === 'analysis' && <AnalysisView datasets={datasets} summaries={summaries} />}
            
            {(activeView === 'charts' || activeView === 'tables' || activeView === 'analysis') && 
             !datasets.case1 && !datasets.case2 && !datasets.excel && (
              <div className="bg-white rounded-lg shadow p-6">
                <div className="text-center py-12">
                  <div className="text-gray-400 mb-4">
                    <BarChart3 className="h-16 w-16 mx-auto" />
                  </div>
                  <h3 className="text-lg font-medium text-gray-900 mb-2">
                    {activeView === 'charts' && 'Interactive Charts'}
                    {activeView === 'tables' && 'Data Tables'}
                    {activeView === 'analysis' && 'Advanced Analysis'}
                  </h3>
                  <p className="text-gray-500">
                    Load the data files in the Overview tab to access this section.
                    {activeView === 'charts' && ' Charts will show profit trends, production comparisons, and inventory analysis.'}
                    {activeView === 'tables' && ' Tables will display detailed data with sorting and filtering capabilities.'}
                    {activeView === 'analysis' && ' Advanced analysis will provide deeper insights and recommendations.'}
                  </p>
                </div>
              </div>
            )}
          </>
        )}
      </div>
    </div>
  );
};

export default DataComparison;