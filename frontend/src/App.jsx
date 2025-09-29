import React, { useState } from 'react'
import DataComparison from './components/DataComparison'
import ChartsView from './components/ChartsView'
import TablesView from './components/TablesView'
import AnalysisView from './components/AnalysisView'
import './App.css'

function App() {
  return (
    <div className="min-h-screen bg-gray-50">
      <DataComparison />
    </div>
  )
}

export default App
