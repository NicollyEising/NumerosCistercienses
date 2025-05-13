import React from 'react';
import logo from './logo.svg';
import Cisterciense from './cisterciense'; // Importando o componente Cisterciense
import 'semantic-ui-css/semantic.min.css';

function App() {
  return (
    <div className="App">
      <header className="App-header">
        <Cisterciense /> {/* Usando o componente Cisterciense */}
      </header>
    </div>
  );
}

export default App;
