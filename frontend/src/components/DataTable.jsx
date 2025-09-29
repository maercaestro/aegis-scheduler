import React, { useState, useMemo } from 'react';
import { ChevronUp, ChevronDown, Search, Filter, Download } from 'lucide-react';

const DataTable = ({ data, title, columns, className = "" }) => {
  const [sortField, setSortField] = useState('');
  const [sortDirection, setSortDirection] = useState('asc');
  const [filterText, setFilterText] = useState('');
  const [currentPage, setCurrentPage] = useState(1);
  const [rowsPerPage] = useState(10);

  // Filter and sort data
  const filteredAndSortedData = useMemo(() => {
    let filtered = data || [];
    
    // Apply text filter
    if (filterText) {
      filtered = filtered.filter(row => 
        Object.values(row).some(value => 
          value && value.toString().toLowerCase().includes(filterText.toLowerCase())
        )
      );
    }
    
    // Apply sorting
    if (sortField) {
      filtered.sort((a, b) => {
        let aVal = a[sortField];
        let bVal = b[sortField];
        
        // Handle numeric values
        if (!isNaN(parseFloat(aVal)) && !isNaN(parseFloat(bVal))) {
          aVal = parseFloat(aVal);
          bVal = parseFloat(bVal);
        } else {
          aVal = aVal ? aVal.toString() : '';
          bVal = bVal ? bVal.toString() : '';
        }
        
        if (aVal < bVal) return sortDirection === 'asc' ? -1 : 1;
        if (aVal > bVal) return sortDirection === 'asc' ? 1 : -1;
        return 0;
      });
    }
    
    return filtered;
  }, [data, filterText, sortField, sortDirection]);

  // Pagination
  const totalPages = Math.ceil(filteredAndSortedData.length / rowsPerPage);
  const startIndex = (currentPage - 1) * rowsPerPage;
  const paginatedData = filteredAndSortedData.slice(startIndex, startIndex + rowsPerPage);

  const handleSort = (field) => {
    if (sortField === field) {
      setSortDirection(sortDirection === 'asc' ? 'desc' : 'asc');
    } else {
      setSortField(field);
      setSortDirection('asc');
    }
  };

  const handleExport = () => {
    if (!filteredAndSortedData.length) return;
    
    const csvContent = [
      columns.map(col => col.header).join(','),
      ...filteredAndSortedData.map(row => 
        columns.map(col => {
          const value = row[col.key] || '';
          return typeof value === 'string' && value.includes(',') 
            ? `"${value}"` 
            : value;
        }).join(',')
      )
    ].join('\n');
    
    const blob = new Blob([csvContent], { type: 'text/csv' });
    const url = window.URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `${title.replace(/\s+/g, '_').toLowerCase()}_export.csv`;
    link.click();
    window.URL.revokeObjectURL(url);
  };

  if (!data || data.length === 0) {
    return (
      <div className={`bg-white rounded-lg shadow ${className}`}>
        <div className="p-6">
          <h3 className="text-lg font-semibold text-gray-800 mb-4">{title}</h3>
          <div className="text-center py-8">
            <p className="text-gray-500">No data available</p>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className={`bg-white rounded-lg shadow ${className}`}>
      <div className="p-6">
        {/* Header */}
        <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between mb-4">
          <h3 className="text-lg font-semibold text-gray-800 mb-2 sm:mb-0">{title}</h3>
          <div className="flex items-center space-x-2">
            <div className="relative">
              <Search className="h-4 w-4 absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400" />
              <input
                type="text"
                placeholder="Search..."
                value={filterText}
                onChange={(e) => {
                  setFilterText(e.target.value);
                  setCurrentPage(1);
                }}
                className="pl-9 pr-3 py-2 border border-gray-300 rounded-md text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
              />
            </div>
            <button
              onClick={handleExport}
              className="flex items-center px-3 py-2 text-sm font-medium text-gray-700 bg-gray-100 hover:bg-gray-200 border border-gray-300 rounded-md transition-colors"
            >
              <Download className="h-4 w-4 mr-1" />
              Export
            </button>
          </div>
        </div>

        {/* Results count */}
        <div className="mb-4 text-sm text-gray-600">
          Showing {paginatedData.length} of {filteredAndSortedData.length} records
          {filteredAndSortedData.length !== data.length && ` (filtered from ${data.length} total)`}
        </div>

        {/* Table */}
        <div className="overflow-x-auto">
          <table className="min-w-full divide-y divide-gray-200">
            <thead className="bg-gray-50">
              <tr>
                {columns.map((column) => (
                  <th
                    key={column.key}
                    onClick={() => handleSort(column.key)}
                    className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider cursor-pointer hover:bg-gray-100 select-none"
                  >
                    <div className="flex items-center">
                      {column.header}
                      <div className="ml-2 flex flex-col">
                        <ChevronUp 
                          className={`h-3 w-3 ${
                            sortField === column.key && sortDirection === 'asc' 
                              ? 'text-blue-500' 
                              : 'text-gray-400'
                          }`} 
                        />
                        <ChevronDown 
                          className={`h-3 w-3 -mt-1 ${
                            sortField === column.key && sortDirection === 'desc' 
                              ? 'text-blue-500' 
                              : 'text-gray-400'
                          }`} 
                        />
                      </div>
                    </div>
                  </th>
                ))}
              </tr>
            </thead>
            <tbody className="bg-white divide-y divide-gray-200">
              {paginatedData.map((row, index) => (
                <tr key={row.id || index} className="hover:bg-gray-50">
                  {columns.map((column) => (
                    <td key={column.key} className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                      {column.render ? column.render(row[column.key], row) : (row[column.key] || '-')}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        {/* Pagination */}
        {totalPages > 1 && (
          <div className="mt-4 flex items-center justify-between">
            <div className="text-sm text-gray-700">
              Page {currentPage} of {totalPages}
            </div>
            <div className="flex space-x-2">
              <button
                onClick={() => setCurrentPage(Math.max(1, currentPage - 1))}
                disabled={currentPage === 1}
                className="px-3 py-2 text-sm font-medium text-gray-700 bg-white border border-gray-300 rounded-md hover:bg-gray-50 disabled:opacity-50 disabled:cursor-not-allowed"
              >
                Previous
              </button>
              <button
                onClick={() => setCurrentPage(Math.min(totalPages, currentPage + 1))}
                disabled={currentPage === totalPages}
                className="px-3 py-2 text-sm font-medium text-gray-700 bg-white border border-gray-300 rounded-md hover:bg-gray-50 disabled:opacity-50 disabled:cursor-not-allowed"
              >
                Next
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

// CSV Data columns configuration
export const csvColumns = [
  { key: 'Date', header: 'Date' },
  { key: 'Slot', header: 'Slot' },
  { key: 'Final Product', header: 'Product' },
  { 
    key: 'Quantity Produced', 
    header: 'Quantity',
    render: (value) => value ? parseFloat(value).toLocaleString(undefined, { maximumFractionDigits: 1 }) : '-'
  },
  { 
    key: 'Profit', 
    header: 'Profit',
    render: (value) => value ? '$' + parseFloat(value).toLocaleString(undefined, { maximumFractionDigits: 0 }) : '-'
  },
  { 
    key: 'Inventory Available', 
    header: 'Inventory',
    render: (value) => value ? parseFloat(value).toLocaleString(undefined, { maximumFractionDigits: 1 }) : '-'
  },
  { 
    key: 'Ullage', 
    header: 'Ullage',
    render: (value) => value ? parseFloat(value).toLocaleString(undefined, { maximumFractionDigits: 1 }) : '-'
  },
];

// Excel Data columns configuration
export const excelColumns = [
  { key: 'Day', header: 'Day' },
  { key: 'Date', header: 'Date' },
  { 
    key: 'Ullage', 
    header: 'Ullage',
    render: (value) => value ? parseFloat(value).toLocaleString(undefined, { maximumFractionDigits: 1 }) : '-'
  },
  { 
    key: 'Processing Inventory', 
    header: 'Processing Inventory',
    render: (value) => value ? parseFloat(value).toLocaleString(undefined, { maximumFractionDigits: 1 }) : '-'
  },
  { 
    key: 'Inventory_Base', 
    header: 'Inv Base',
    render: (value) => value ? parseFloat(value).toLocaleString(undefined, { maximumFractionDigits: 1 }) : '-'
  },
  { 
    key: 'Inventory_A', 
    header: 'Inv A',
    render: (value) => value ? parseFloat(value).toLocaleString(undefined, { maximumFractionDigits: 1 }) : '-'
  },
  { 
    key: 'Inventory_B', 
    header: 'Inv B',
    render: (value) => value ? parseFloat(value).toLocaleString(undefined, { maximumFractionDigits: 1 }) : '-'
  },
  { key: 'primary_grade', header: 'Primary Grade' },
  { key: 'secondary_grade', header: 'Secondary Grade' },
];

// Crude Oil columns for CSV data
export const crudeOilColumns = [
  { key: 'Date', header: 'Date' },
  { 
    key: 'Crude Tapis Available', 
    header: 'Tapis Available',
    render: (value) => value ? parseFloat(value).toLocaleString(undefined, { maximumFractionDigits: 1 }) : '-'
  },
  { 
    key: 'Crude Minas Available', 
    header: 'Minas Available',
    render: (value) => value ? parseFloat(value).toLocaleString(undefined, { maximumFractionDigits: 1 }) : '-'
  },
  { 
    key: 'Crude Sepat Available', 
    header: 'Sepat Available',
    render: (value) => value ? parseFloat(value).toLocaleString(undefined, { maximumFractionDigits: 1 }) : '-'
  },
  { 
    key: 'Crude Kimanis Available', 
    header: 'Kimanis Available',
    render: (value) => value ? parseFloat(value).toLocaleString(undefined, { maximumFractionDigits: 1 }) : '-'
  },
  { 
    key: 'Crude KIMC Available', 
    header: 'KIMC Available',
    render: (value) => value ? parseFloat(value).toLocaleString(undefined, { maximumFractionDigits: 1 }) : '-'
  },
  { 
    key: 'Crude Bintulu Available', 
    header: 'Bintulu Available',
    render: (value) => value ? parseFloat(value).toLocaleString(undefined, { maximumFractionDigits: 1 }) : '-'
  },
];

export default DataTable;