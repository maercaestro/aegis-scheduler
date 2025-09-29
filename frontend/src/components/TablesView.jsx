import React from 'react';
import DataTable, { csvColumns, excelColumns, crudeOilColumns } from './DataTable';

const TablesView = ({ datasets }) => {
  const { case1, case2, excel } = datasets;

  return (
    <div className="space-y-6">
      {/* Main Data Tables */}
      <div className="grid grid-cols-1 xl:grid-cols-2 gap-6">
        {case1 && (
          <DataTable
            data={case1}
            title="Case 1 Data"
            columns={csvColumns}
            className="border-l-4 border-blue-500"
          />
        )}
        
        {case2 && (
          <DataTable
            data={case2}
            title="Case 2 Data"
            columns={csvColumns}
            className="border-l-4 border-red-500"
          />
        )}
      </div>

      {excel && (
        <DataTable
          data={excel}
          title="Base Case (Excel) Data"
          columns={excelColumns}
          className="border-l-4 border-green-500"
        />
      )}

      {/* Crude Oil Data Tables */}
      {(case1 || case2) && (
        <div className="bg-white rounded-lg shadow p-6">
          <h2 className="text-xl font-semibold text-gray-800 mb-6">Crude Oil Availability</h2>
          <div className="grid grid-cols-1 xl:grid-cols-2 gap-6">
            {case1 && (
              <DataTable
                data={case1}
                title="Case 1 - Crude Oil Data"
                columns={crudeOilColumns}
                className="border-l-4 border-blue-500"
              />
            )}
            
            {case2 && (
              <DataTable
                data={case2}
                title="Case 2 - Crude Oil Data"
                columns={crudeOilColumns}
                className="border-l-4 border-red-500"
              />
            )}
          </div>
        </div>
      )}
    </div>
  );
};

export default TablesView;