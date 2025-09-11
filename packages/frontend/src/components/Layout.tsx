import { ReactNode } from 'react';
import { Link, useLocation } from 'react-router-dom';
import { 
  Monitor, 
  Cpu, 
  MessageSquare, 
  Settings,
  Activity,
  Wifi,
  WifiOff
} from 'lucide-react';
import { useWebSocket } from '../hooks/useWebSocket';
import clsx from 'clsx';

interface LayoutProps {
  children: ReactNode;
}

export function Layout({ children }: LayoutProps) {
  const location = useLocation();
  const { isConnected } = useWebSocket();

  const navigation = [
    { name: 'Dashboard', href: '/', icon: Monitor },
    { name: 'Devices', href: '/devices', icon: Cpu },
    { name: 'Models', href: '/models', icon: Settings },
    { name: 'Chat', href: '/chat', icon: MessageSquare },
  ];

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Sidebar */}
      <div className="fixed inset-y-0 left-0 z-50 w-64 bg-white shadow-sm">
        <div className="flex h-16 shrink-0 items-center border-b border-gray-200 px-6">
          <h1 className="text-xl font-semibold text-gray-900">Orchard</h1>
          <div className="ml-auto">
            {isConnected ? (
              <div className="flex items-center text-green-600">
                <Wifi className="h-4 w-4" />
                <span className="ml-1 text-xs">Connected</span>
              </div>
            ) : (
              <div className="flex items-center text-red-600">
                <WifiOff className="h-4 w-4" />
                <span className="ml-1 text-xs">Disconnected</span>
              </div>
            )}
          </div>
        </div>
        
        <nav className="flex flex-1 flex-col px-4 py-4">
          <ul role="list" className="space-y-1">
            {navigation.map((item) => (
              <li key={item.name}>
                <Link
                  to={item.href}
                  className={clsx(
                    location.pathname === item.href
                      ? 'bg-primary-50 text-primary-700'
                      : 'text-gray-700 hover:bg-gray-50',
                    'group flex gap-x-3 rounded-md p-2 text-sm leading-6 font-semibold'
                  )}
                >
                  <item.icon
                    className={clsx(
                      location.pathname === item.href
                        ? 'text-primary-700'
                        : 'text-gray-400 group-hover:text-gray-600',
                      'h-6 w-6 shrink-0'
                    )}
                  />
                  {item.name}
                </Link>
              </li>
            ))}
          </ul>
        </nav>
      </div>

      {/* Main content */}
      <div className="pl-64">
        <div className="flex h-16 shrink-0 items-center gap-x-4 border-b border-gray-200 bg-white px-4 shadow-sm sm:gap-x-6 sm:px-6 lg:px-8">
          <div className="flex flex-1 gap-x-4 self-stretch lg:gap-x-6">
            <div className="flex items-center gap-x-4 lg:gap-x-6">
              <Activity className="h-5 w-5 text-gray-400" />
              <span className="text-sm text-gray-600">
                Distributed LLM Platform
              </span>
            </div>
          </div>
        </div>

        <main className="py-6">
          <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
            {children}
          </div>
        </main>
      </div>
    </div>
  );
} 
