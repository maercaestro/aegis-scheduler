import * as XLSX from 'xlsx';
import Papa from 'papaparse';

/**
 * Data loading and processing utilities for the comparison app
 */

// Helper function to normalize dates
export const normalizeDate = (dateValue) => {
  if (!dateValue) return null;
  
  if (dateValue instanceof Date) {
    return dateValue.toISOString().split('T')[0];
  }
  
  if (typeof dateValue === 'string') {
    const date = new Date(dateValue);
    return isNaN(date) ? dateValue : date.toISOString().split('T')[0];
  }
  
  return dateValue.toString();
};

// Helper function to convert day number to date (assuming start date)
export const dayToDate = (dayNum, startDate = '2025-10-01') => {
  const start = new Date(startDate);
  start.setDate(start.getDate() + dayNum);
  return start.toISOString().split('T')[0];
};

/**
 * Load CSV file data
 */
export const loadCSV = (file) => {
  return new Promise((resolve, reject) => {
    Papa.parse(file, {
      header: true,
      skipEmptyLines: true,
      complete: (results) => {
        if (results.errors.length > 0) {
          console.warn('CSV parsing warnings:', results.errors);
        }
        
        // Process the data
        const processedData = results.data.map((row, index) => ({
          ...row,
          id: index,
          Date: normalizeDate(row.Date),
          'Quantity Produced': parseFloat(row['Quantity Produced']) || 0,
          Profit: parseFloat(row.Profit) || 0,
          'Inventory Available': parseFloat(row['Inventory Available']) || 0,
          Ullage: parseFloat(row.Ullage) || 0,
          // Parse crude oil data
          'Crude Tapis Available': parseFloat(row['Crude Tapis Available']) || 0,
          'Crude Minas Available': parseFloat(row['Crude Minas Available']) || 0,
          'Crude Sepat Available': parseFloat(row['Crude Sepat Available']) || 0,
          'Crude Kimanis Available': parseFloat(row['Crude Kimanis Available']) || 0,
          'Crude KIMC Available': parseFloat(row['Crude KIMC Available']) || 0,
          'Crude Bintulu Available': parseFloat(row['Crude Bintulu Available']) || 0,
        }));
        
        resolve(processedData);
      },
      error: (error) => {
        reject(error);
      }
    });
  });
};

/**
 * Load Excel file data from specific sheet
 */
export const loadExcel = (file, sheetName = 'Base Case (Working)') => {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = (e) => {
      try {
        const data = new Uint8Array(e.target.result);
        const workbook = XLSX.read(data, { type: 'array' });
        
        if (!workbook.SheetNames.includes(sheetName)) {
          reject(new Error(`Sheet "${sheetName}" not found. Available sheets: ${workbook.SheetNames.join(', ')}`));
          return;
        }
        
        const worksheet = workbook.Sheets[sheetName];
        const jsonData = XLSX.utils.sheet_to_json(worksheet, { header: 1 });
        
        // Find header row (usually row 1 or 2)
        let headerRow = -1;
        for (let i = 0; i < Math.min(5, jsonData.length); i++) {
          if (jsonData[i] && jsonData[i].includes('Day')) {
            headerRow = i;
            break;
          }
        }
        
        if (headerRow === -1) {
          reject(new Error('Could not find header row with "Day" column'));
          return;
        }
        
        const headers = jsonData[headerRow];
        const dataRows = jsonData.slice(headerRow + 1);
        
        // Process the data
        const processedData = dataRows
          .filter(row => row && row.length > 0 && row[0] !== null && row[0] !== undefined)
          .map((row, index) => {
            const rowObj = { id: index };
            headers.forEach((header, colIndex) => {
              if (header) {
                rowObj[header] = row[colIndex];
              }
            });
            
            // Normalize data types
            if (rowObj.Day !== undefined) {
              rowObj.Day = parseFloat(rowObj.Day);
              rowObj.Date = dayToDate(rowObj.Day);
            }
            if (rowObj.Ullage !== undefined) rowObj.Ullage = parseFloat(rowObj.Ullage) || 0;
            if (rowObj['Processing Inventory'] !== undefined) {
              rowObj['Processing Inventory'] = parseFloat(rowObj['Processing Inventory']) || 0;
            }
            
            // Parse inventory columns
            ['Inventory_Base', 'Inventory_A', 'Inventory_B', 'Inventory_C', 'Inventory_D', 'Inventory_E', 'Inventory_F'].forEach(col => {
              if (rowObj[col] !== undefined) {
                rowObj[col] = parseFloat(rowObj[col]) || 0;
              }
            });
            
            // Parse rate columns
            ['Rate_Base', 'Rate_A', 'Rate_B', 'Rate_C', 'Rate_D', 'Rate_E', 'Rate_F'].forEach(col => {
              if (rowObj[col] !== undefined) {
                rowObj[col] = parseFloat(rowObj[col]) || 0;
              }
            });
            
            return rowObj;
          });
        
        resolve(processedData);
      } catch (error) {
        reject(error);
      }
    };
    reader.onerror = () => reject(new Error('Failed to read file'));
    reader.readAsArrayBuffer(file);
  });
};

/**
 * Calculate summary statistics for a dataset
 */
export const calculateSummary = (data, type) => {
  if (!data || data.length === 0) return null;
  
  const summary = {
    totalRows: data.length,
    type: type
  };
  
  if (type === 'csv') {
    // CSV-specific metrics
    const totalProfit = data.reduce((sum, row) => sum + (parseFloat(row.Profit) || 0), 0);
    const totalProduction = data.reduce((sum, row) => sum + (parseFloat(row['Quantity Produced']) || 0), 0);
    const products = [...new Set(data.map(row => row['Final Product']).filter(Boolean))];
    const dateRange = {
      start: data.length > 0 ? data[0].Date : null,
      end: data.length > 0 ? data[data.length - 1].Date : null
    };
    
    summary.totalProfit = totalProfit;
    summary.totalProduction = totalProduction;
    summary.avgProfit = totalProfit / data.length;
    summary.avgProduction = totalProduction / data.length;
    summary.products = products;
    summary.dateRange = dateRange;
    
    // Inventory statistics
    const inventoryData = data.map(row => parseFloat(row['Inventory Available']) || 0).filter(val => val > 0);
    if (inventoryData.length > 0) {
      summary.avgInventory = inventoryData.reduce((a, b) => a + b, 0) / inventoryData.length;
      summary.maxInventory = Math.max(...inventoryData);
      summary.minInventory = Math.min(...inventoryData);
    }
    
  } else if (type === 'excel') {
    // Excel-specific metrics
    const dayRange = {
      start: Math.min(...data.map(row => row.Day || 0)),
      end: Math.max(...data.map(row => row.Day || 0))
    };
    
    summary.dayRange = dayRange;
    
    // Processing inventory statistics
    const procInventory = data.map(row => parseFloat(row['Processing Inventory']) || 0).filter(val => val > 0);
    if (procInventory.length > 0) {
      summary.avgProcessingInventory = procInventory.reduce((a, b) => a + b, 0) / procInventory.length;
      summary.maxProcessingInventory = Math.max(...procInventory);
      summary.minProcessingInventory = Math.min(...procInventory);
    }
    
    // Ullage statistics
    const ullageData = data.map(row => parseFloat(row.Ullage) || 0).filter(val => val > 0);
    if (ullageData.length > 0) {
      summary.avgUllage = ullageData.reduce((a, b) => a + b, 0) / ullageData.length;
      summary.maxUllage = Math.max(...ullageData);
      summary.minUllage = Math.min(...ullageData);
    }
    
    // Count different inventory types
    const inventoryTypes = ['Inventory_Base', 'Inventory_A', 'Inventory_B', 'Inventory_C', 'Inventory_D', 'Inventory_E', 'Inventory_F'];
    summary.inventoryBreakdown = {};
    inventoryTypes.forEach(type => {
      const values = data.map(row => parseFloat(row[type]) || 0);
      const total = values.reduce((a, b) => a + b, 0);
      if (total > 0) {
        summary.inventoryBreakdown[type] = {
          total,
          avg: total / values.length,
          max: Math.max(...values)
        };
      }
    });
  }
  
  return summary;
};

/**
 * Compare two datasets and return comparison metrics
 */
export const compareDatasets = (data1, data2, data3) => {
  const summaries = [
    calculateSummary(data1, 'csv'),
    calculateSummary(data2, 'csv'), 
    calculateSummary(data3, 'excel')
  ];
  
  return {
    summaries,
    comparison: {
      csv1VsCsv2: {
        profitDiff: summaries[1]?.totalProfit - summaries[0]?.totalProfit,
        productionDiff: summaries[1]?.totalProduction - summaries[0]?.totalProduction,
        profitPctDiff: summaries[0]?.totalProfit ? ((summaries[1]?.totalProfit - summaries[0]?.totalProfit) / summaries[0]?.totalProfit) * 100 : 0
      }
    }
  };
};

/**
 * Prepare data for charting
 */
export const prepareChartData = (datasets, chartType) => {
  const [csv1, csv2, excel] = datasets;
  
  switch (chartType) {
    case 'profit_comparison':
      return {
        labels: ['Case 1', 'Case 2', 'Base Case (Excel)'],
        datasets: [{
          label: 'Total Profit',
          data: [
            csv1?.reduce((sum, row) => sum + (parseFloat(row.Profit) || 0), 0) || 0,
            csv2?.reduce((sum, row) => sum + (parseFloat(row.Profit) || 0), 0) || 0,
            0 // Excel doesn't have direct profit data
          ],
          backgroundColor: ['#3B82F6', '#EF4444', '#10B981'],
        }]
      };
      
    case 'production_comparison':
      return {
        labels: ['Case 1', 'Case 2', 'Base Case (Excel)'],
        datasets: [{
          label: 'Total Production',
          data: [
            csv1?.reduce((sum, row) => sum + (parseFloat(row['Quantity Produced']) || 0), 0) || 0,
            csv2?.reduce((sum, row) => sum + (parseFloat(row['Quantity Produced']) || 0), 0) || 0,
            excel?.reduce((sum, row) => sum + (parseFloat(row['Processing Inventory']) || 0), 0) || 0
          ],
          backgroundColor: ['#3B82F6', '#EF4444', '#10B981'],
        }]
      };
      
    case 'inventory_trend':
      // Create time series for inventory
      const csv1Inventory = csv1?.map(row => ({ 
        x: row.Date, 
        y: parseFloat(row['Inventory Available']) || 0 
      })) || [];
      const csv2Inventory = csv2?.map(row => ({ 
        x: row.Date, 
        y: parseFloat(row['Inventory Available']) || 0 
      })) || [];
      const excelInventory = excel?.map(row => ({ 
        x: row.Date, 
        y: parseFloat(row['Processing Inventory']) || 0 
      })) || [];
      
      return {
        datasets: [
          {
            label: 'Case 1 Inventory',
            data: csv1Inventory,
            borderColor: '#3B82F6',
            backgroundColor: '#3B82F640',
          },
          {
            label: 'Case 2 Inventory',
            data: csv2Inventory,
            borderColor: '#EF4444',
            backgroundColor: '#EF444440',
          },
          {
            label: 'Base Case Processing Inventory',
            data: excelInventory,
            borderColor: '#10B981',
            backgroundColor: '#10B98140',
          }
        ]
      };
      
    default:
      return { labels: [], datasets: [] };
  }
};