import { useState } from 'react';

// Componente funcional Sidebar
function Sidebar() {
    // Estado para controlar si el sidebar está abierto o cerrado
    const [isOpen, setIsOpen] = useState(true);

    // Función para alternar la visibilidad del sidebar
    const toggleSidebar = () => {
        setIsOpen(!isOpen);
    };

    return (
        <>
            {/* Botón para mostrar/ocultar el sidebar */}
            <button 
                onClick={toggleSidebar}
                className={`fixed top-4 ${isOpen ? 'left-[13rem]' : 'left-4'} z-50 inline-flex items-center justify-center w-8 h-8 text-gray-500 bg-white rounded-full shadow-md hover:bg-gray-100 focus:outline-none focus:ring-2 focus:ring-gray-200 dark:bg-gray-800 dark:text-gray-400 dark:hover:bg-gray-700 dark:focus:ring-gray-600 transition-all duration-300`}
            >
                <span className="sr-only">Toggle sidebar</span>
                {/* Icono de flechas para indicar abrir/cerrar */}
                <svg 
                    className={`w-5 h-5 transition-transform duration-300 ${isOpen ? 'rotate-180' : ''}`} 
                    fill="none" 
                    stroke="currentColor" 
                    viewBox="0 0 24 24"
                >
                    <path 
                        strokeLinecap="round" 
                        strokeLinejoin="round" 
                        strokeWidth="2" 
                        d="M11 19l-7-7 7-7m8 14l-7-7 7-7"
                    />
                </svg>
            </button>

            {/* Sidebar lateral */}
            <aside 
                className={`fixed top-0 left-0 z-40 w-56 h-screen transition-transform duration-300 ease-in-out ${isOpen ? 'translate-x-0' : '-translate-x-full'} shadow-lg`}
                aria-label="Sidebar"
            >
                <div className="h-full px-2 py-3 overflow-y-auto bg-white dark:bg-gray-800 border-r border-gray-200 dark:border-gray-700">
                    {/* Logo y título */}
                    <div className="flex items-center justify-center mb-6">
                        <img src="https://flowbite.com/docs/images/logo.svg" className="h-6 me-2" alt="Logo" />
                        <span className="text-lg font-bold text-gray-800 dark:text-white">Dev Senior</span>
                    </div>
                    
                    {/* Menú de navegación */}
                    <ul className="space-y-2 font-medium">
                        {/* Opción Dashboard */}
                        <li>
                            <a href="#" className="flex items-center p-2 text-gray-900 rounded-lg dark:text-white hover:bg-blue-50 dark:hover:bg-gray-700 group transition-all duration-200">
                                {/* Icono Dashboard */}
                                <svg className="w-5 h-5 text-blue-600 dark:text-blue-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M3 12l2-2m0 0l7-7 7 7M5 10v10a1 1 0 001 1h3m10-11l2 2m-2-2v10a1 1 0 01-1 1h-3m-6 0a1 1 0 001-1v-4a1 1 0 011-1h2a1 1 0 011 1v4a1 1 0 001 1m-6 0h6"/>
                                </svg>
                                <span className="ml-2 text-sm">Dashboard</span>
                            </a>
                        </li>
                        {/* Opción Proyectos */}
                        <li>
                            <a href="#" className="flex items-center p-2 text-gray-900 rounded-lg dark:text-white hover:bg-blue-50 dark:hover:bg-gray-700 group transition-all duration-200">
                                {/* Icono Proyectos */}
                                <svg className="w-5 h-5 text-blue-600 dark:text-blue-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M3 7v10a2 2 0 002 2h14a2 2 0 002-2V9a2 2 0 00-2-2h-6l-2-2H5a2 2 0 00-2 2z"/>
                                </svg>
                                <span className="ml-2 text-sm">Proyectos</span>
                            </a>
                        </li>
                        {/* Opción Modelos de IA */}
                        <li>
                            <a href="#" className="flex items-center p-2 text-gray-900 rounded-lg dark:text-white hover:bg-blue-50 dark:hover:bg-gray-700 group transition-all duration-200">
                                {/* Icono Modelos de IA */}
                                <svg className="w-5 h-5 text-blue-600 dark:text-blue-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9 3v2m6-2v2M9 19v2m6-2v2M5 9H3m2 6H3m18-6h-2m2 6h-2M7 19h10a2 2 0 002-2V7a2 2 0 00-2-2H7a2 2 0 00-2 2v10a2 2 0 002 2zM9 9h6v6H9V9z"/>
                                </svg>
                                <span className="ml-2 text-sm">Modelos de IA</span>
                            </a>
                        </li>
                        {/* Opción Datos */}
                        <li>
                            <a href="#" className="flex items-center p-2 text-gray-900 rounded-lg dark:text-white hover:bg-blue-50 dark:hover:bg-gray-700 group transition-all duration-200">
                                {/* Icono Datos */}
                                <svg className="w-5 h-5 text-blue-600 dark:text-blue-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M4 7v10c0 2.21 3.582 4 8 4s8-1.79 8-4V7M4 7c0 2.21 3.582 4 8 4s8-1.79 8-4M4 7c0-2.21 3.582-4 8-4s8 1.79 8 4m0 5c0 2.21-3.582 4-8 4s-8-1.79-8-4"/>
                                </svg>
                                <span className="ml-2 text-sm">Datos</span>
                            </a>
                        </li>
                        {/* Opción Configuración */}
                        <li>
                            <a href="#" className="flex items-center p-2 text-gray-900 rounded-lg dark:text-white hover:bg-blue-50 dark:hover:bg-gray-700 group transition-all duration-200">
                                {/* Icono Configuración */}
                                <svg className="w-5 h-5 text-blue-600 dark:text-blue-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z"/>
                                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M15 12a3 3 0 11-6 0 3 3 0 016 0z"/>
                                </svg>
                                <span className="ml-2 text-sm">Configuración</span>
                            </a>
                        </li>
                    </ul>
                </div>
            </aside>
        </>
    );
}

export default Sidebar;