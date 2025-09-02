import React, { useState, useEffect, useRef } from 'react';
import { MapContainer, TileLayer, Marker, Popup, useMap } from 'react-leaflet';
import { MapPin, Phone, Clock, Navigation, AlertTriangle, Guitar as Hospital, MapPinIcon, Loader } from 'lucide-react';
import L from 'leaflet';
import 'leaflet/dist/leaflet.css';

// Fix default Leaflet marker icons
delete (L.Icon.Default.prototype as any)._getIconUrl;
L.Icon.Default.mergeOptions({
  iconRetinaUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-icon-2x.png',
  iconUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-icon.png',
  shadowUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-shadow.png',
});

// Create red marker icon for emergency services
const redIcon = new L.Icon({
  iconUrl: 'https://raw.githubusercontent.com/pointhi/leaflet-color-markers/master/img/marker-icon-2x-red.png',
  shadowUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.3.1/images/marker-shadow.png',
  iconSize: [25, 41],
  iconAnchor: [12, 41],
  popupAnchor: [1, -34],
  shadowSize: [41, 41]
});

// Component to change map view when user location changes
// ChangeView component removed to allow free map navigation

interface EmergencyService {
  id: string;
  name: string;
  description: string;
  phone: string;
  category: string;
  coordinates: [number, number]; // [lat, lng]
  distance?: number;
}

interface UserLocation {
  lat: number;
  lng: number;
}

// Haversine formula to calculate distance between two points
const calculateDistance = (lat1: number, lon1: number, lat2: number, lon2: number): number => {
  const R = 6371; // Radius of the Earth in kilometers
  const dLat = (lat2 - lat1) * Math.PI / 180;
  const dLon = (lon2 - lon1) * Math.PI / 180;
  const a = 
    Math.sin(dLat/2) * Math.sin(dLat/2) +
    Math.cos(lat1 * Math.PI / 180) * Math.cos(lat2 * Math.PI / 180) * 
    Math.sin(dLon/2) * Math.sin(dLon/2);
  const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1-a));
  return R * c;
};

// Parse KML data
const parseKMLData = async (): Promise<EmergencyService[]> => {
  try {
    const response = await fetch('/src/Malaysia EMS Map.kml');
    const kmlText = await response.text();
    const parser = new DOMParser();
    const kmlDoc = parser.parseFromString(kmlText, 'text/xml');
    
    const placemarks = kmlDoc.getElementsByTagName('Placemark');
    const services: EmergencyService[] = [];
    
    for (let i = 0; i < placemarks.length; i++) {
      const placemark = placemarks[i];
      const nameElement = placemark.getElementsByTagName('name')[0];
      const descElement = placemark.getElementsByTagName('description')[0];
      const coordElement = placemark.getElementsByTagName('coordinates')[0];
      
      if (nameElement && coordElement) {
        const name = nameElement.textContent?.trim() || '';
        const description = descElement?.textContent || '';
        const coordText = coordElement.textContent?.trim() || '';
        const [lng, lat] = coordText.split(',').map(Number);
        
        if (!isNaN(lat) && !isNaN(lng)) {
          // Extract phone number from description
          const phoneMatch = description.match(/\+60[\s\d\-]+/);
          const phone = phoneMatch ? phoneMatch[0] : '999';
          
          // Determine category based on folder or name
          let category = 'Emergency Service';
          if (name.includes('MRC') || name.includes('Red Crescent')) {
            category = 'Red Crescent Malaysia';
          } else if (name.includes('SJAM') || name.includes('St. John')) {
            category = 'St. John Ambulance Malaysia';
          } else if (name.includes('Civil Defense') || name.includes('Angkatan')) {
            category = 'Civil Defense Malaysia';
          }
          
          services.push({
            id: `service-${i}`,
            name,
            description,
            phone,
            category,
            coordinates: [lat, lng]
          });
        }
      }
    }
    
    return services;
  } catch (error) {
    console.error('Error parsing KML:', error);
    return [];
  }
};



export default function EmergencyMap() {
  const [selectedService, setSelectedService] = useState<EmergencyService | null>(null);
  const [userLocation, setUserLocation] = useState<UserLocation | null>(null);
  const [emergencyServices, setEmergencyServices] = useState<EmergencyService[]>([]);
  const [nearestServices, setNearestServices] = useState<EmergencyService[]>([]);
  const [loading, setLoading] = useState(true);
  const [locationError, setLocationError] = useState<string | null>(null);
  const mapRef = useRef<L.Map | null>(null);

  // Load KML data and get user location on component mount
  useEffect(() => {
    const initializeMap = async () => {
      try {
        // Load KML data
        const services = await parseKMLData();
        setEmergencyServices(services);
        
        // Get user location
        if (navigator.geolocation) {
          navigator.geolocation.getCurrentPosition(
            (position) => {
              const userLoc = {
                lat: position.coords.latitude,
                lng: position.coords.longitude
              };
              setUserLocation(userLoc);
              
              // Calculate distances and get nearest services
              const servicesWithDistance = services.map(service => ({
                ...service,
                distance: calculateDistance(
                  userLoc.lat,
                  userLoc.lng,
                  service.coordinates[0],
                  service.coordinates[1]
                )
              }));
              
              // Sort by distance and get top 5
              const nearest = servicesWithDistance
                .sort((a, b) => (a.distance || 0) - (b.distance || 0))
                .slice(0, 5);
              
              setNearestServices(nearest);
              setLoading(false);
            },
            (error) => {
              console.error('Geolocation error:', error);
              let errorMessage = 'Unable to get your location.';
              
              switch (error.code) {
                case error.PERMISSION_DENIED:
                  errorMessage = 'Location access denied. Please enable location services in your browser settings and refresh the page for accurate emergency service recommendations.';
                  break;
                case error.POSITION_UNAVAILABLE:
                  errorMessage = 'Location information is unavailable. Using default location (Kuala Lumpur).';
                  break;
                case error.TIMEOUT:
                  errorMessage = 'Location request timed out. Using default location (Kuala Lumpur).';
                  break;
                default:
                  errorMessage = 'An unknown error occurred while retrieving location. Using default location (Kuala Lumpur).';
                  break;
              }
              
              setLocationError(errorMessage);
              // Default to Kuala Lumpur coordinates
              setUserLocation({ lat: 3.139, lng: 101.6869 });
              setNearestServices(services.slice(0, 5));
              setLoading(false);
            },
            {
              enableHighAccuracy: true,
              timeout: 10000,
              maximumAge: 300000
            }
          );
        } else {
          setLocationError('Geolocation is not supported by this browser.');
          setUserLocation({ lat: 3.139, lng: 101.6869 });
          setNearestServices(services.slice(0, 5));
          setLoading(false);
        }
      } catch (error) {
        console.error('Error initializing map:', error);
        setLoading(false);
      }
    };
    
    initializeMap();
  }, []);

  const handleEmergencyCall = (service: EmergencyService) => {
    if (service.phone) {
      window.open(`tel:${service.phone}`, '_self');
    } else {
      window.open('tel:999', '_self');
    }
  };

  const getCategoryColor = (category: string) => {
    switch (category) {
      case 'Red Crescent Malaysia': return 'text-red-600 bg-red-100';
      case 'St. John Ambulance Malaysia': return 'text-blue-600 bg-blue-100';
      case 'Civil Defense Malaysia': return 'text-green-600 bg-green-100';
      default: return 'text-gray-600 bg-gray-100';
    }
  };

  // Default center (Kuala Lumpur)
  const defaultCenter: [number, number] = [3.139, 101.6869];
  const mapCenter = userLocation ? [userLocation.lat, userLocation.lng] as [number, number] : defaultCenter;

  if (loading) {
    return (
      <div className="p-6 space-y-6">
        <div className="bg-gradient-to-r from-red-600 to-orange-600 text-white p-6 rounded-lg">
          <h2 className="text-2xl font-bold mb-2 flex items-center">
            <Hospital className="w-8 h-8 mr-3" />
            Emergency Response Map
          </h2>
          <p className="text-red-100">Loading emergency services and getting your location...</p>
        </div>
        <div className="flex items-center justify-center h-96">
          <Loader className="w-8 h-8 animate-spin text-blue-600" />
          <span className="ml-2 text-gray-600">Loading map data...</span>
        </div>
      </div>
    );
  }

  return (
    <div className="p-6 space-y-6">
      {/* Header */}
      <div className="bg-gradient-to-r from-red-600 to-orange-600 text-white p-6 rounded-lg">
        <h2 className="text-2xl font-bold mb-2 flex items-center">
          <Hospital className="w-8 h-8 mr-3" />
          Emergency Response Map
        </h2>
        <p className="text-red-100">
          {userLocation 
            ? `Showing nearest emergency services based on your location` 
            : 'Emergency services in Malaysia'}
        </p>
        {locationError && (
          <p className="text-yellow-200 text-sm mt-2">{locationError}</p>
        )}
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Map View */}
        <div className="lg:col-span-2">
          <div className="bg-white rounded-lg shadow-lg border border-gray-200 p-6">
            <h3 className="text-lg font-semibold mb-4 flex items-center">
              <MapPin className="w-5 h-5 mr-2" />
              Interactive Map - Emergency Services
            </h3>

            <div className="rounded-lg overflow-hidden h-96 relative">
              <MapContainer
                center={mapCenter}
                zoom={18}
                maxZoom={18}
                style={{ height: '100%', width: '100%' }}
                ref={mapRef}
              >
                <TileLayer
                  url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
                  attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
                />

                
                {/* Removed ChangeView component to allow free map navigation */}
                
                {/* User location marker */}
                {userLocation && (
                  <Marker position={[userLocation.lat, userLocation.lng]}>
                    <Popup>
                      <div className="text-center">
                        <strong>Your Location</strong>
                      </div>
                    </Popup>
                  </Marker>
                )}
                
                {/* Emergency service markers */}
                {nearestServices.map((service) => (
                  <Marker
                    key={service.id}
                    position={service.coordinates}
                    icon={redIcon}
                    eventHandlers={{
                      click: () => setSelectedService(service)
                    }}
                  >
                    <Popup>
                      <div className="max-w-xs">
                        <h4 className="font-semibold text-sm mb-2">{service.name}</h4>
                        <p className="text-xs text-gray-600 mb-2">{service.category}</p>
                        {service.distance && (
                          <p className="text-xs text-blue-600 mb-2">
                            Distance: {service.distance.toFixed(2)} km
                          </p>
                        )}
                        <button
                          onClick={() => handleEmergencyCall(service)}
                          className="w-full bg-red-600 hover:bg-red-700 text-white py-1 px-2 rounded text-xs"
                        >
                          Call {service.phone}
                        </button>
                      </div>
                    </Popup>
                  </Marker>
                ))}
              </MapContainer>
              
              {/* Floating Recenter Button */}
              <button 
                onClick={() => {
                  if (userLocation && mapRef.current) {
                    mapRef.current.setView([userLocation.lat, userLocation.lng], 13);
                  }
                }}
                className="absolute bottom-4 right-4 bg-white hover:bg-gray-50 text-gray-700 p-3 rounded-full shadow-lg border border-gray-200 transition-all hover:shadow-xl z-[1000]"
                title="Recenter to your location"
              >
                <Navigation className="w-5 h-5" />
              </button>
            </div>
          </div>
        </div>

        {/* Emergency Services List */}
        <div className="space-y-4">
          {/* Quick Actions */}
          <div className="bg-white rounded-lg shadow-lg border border-gray-200 p-6">
            <h3 className="text-lg font-semibold mb-4 flex items-center">
              <AlertTriangle className="w-5 h-5 mr-2 text-red-600" />
              Quick Actions
            </h3>
            
            <div className="space-y-3">
              <button 
                 onClick={() => window.open('tel:999', '_self')}
                 className="w-full flex items-center justify-center space-x-2 bg-red-600 hover:bg-red-700 text-white py-3 px-4 rounded-lg font-semibold transition-colors"
               >
                 <Phone className="w-5 h-5" />
                 <span>Call 999 (Emergency)</span>
               </button>
               
               <button 
                 onClick={() => {
                   if (nearestServices.length > 0) {
                     handleEmergencyCall(nearestServices[0]);
                   }
                 }}
                 className="w-full flex items-center justify-center space-x-2 bg-blue-600 hover:bg-blue-700 text-white py-3 px-4 rounded-lg font-semibold transition-colors"
               >
                 <AlertTriangle className="w-5 h-5" />
                 <span>Call Nearest Service</span>
               </button>
            </div>
          </div>

          <div className="bg-white rounded-lg shadow-lg border border-gray-200 p-6">
            <h3 className="text-lg font-semibold mb-4">Top 5 Nearest Emergency Services</h3>
            
            <div className="space-y-4">
              {nearestServices.map((service, index) => (
                <div
                  key={service.id}
                  className={`p-4 border rounded-lg cursor-pointer transition-all ${
                    selectedService?.id === service.id 
                      ? 'border-blue-500 bg-blue-50' 
                      : 'border-gray-200 hover:border-gray-300'
                  }`}
                  onClick={() => setSelectedService(service)}
                >
                  <div className="flex items-start justify-between mb-2">
                    <div className="flex items-center space-x-2">
                      <span className="bg-blue-600 text-white text-xs font-bold rounded-full w-6 h-6 flex items-center justify-center">
                        {index + 1}
                      </span>
                      <h4 className="font-semibold text-sm">{service.name}</h4>
                    </div>
                    <span className={`px-2 py-1 rounded-full text-xs font-medium ${getCategoryColor(service.category)}`}>
                      {service.category.replace(' Malaysia', '')}
                    </span>
                  </div>
                  
                  <div className="space-y-1 text-xs text-gray-600 ml-8">
                    {service.distance && (
                      <div className="flex items-center space-x-2">
                        <Navigation className="w-3 h-3" />
                        <span>{service.distance.toFixed(2)} km away</span>
                      </div>
                    )}
                    <div className="flex items-center space-x-2">
                      <Phone className="w-3 h-3" />
                      <span>{service.phone}</span>
                    </div>
                  </div>

                  {service.description && (
                    <div className="mt-2 ml-8">
                      <p className="text-xs text-gray-500 line-clamp-2">
                        {service.description.replace(/<[^>]*>/g, '').substring(0, 100)}...
                      </p>
                    </div>
                  )}

                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      handleEmergencyCall(service);
                    }}
                    className="w-full mt-3 flex items-center justify-center space-x-2 bg-red-600 hover:bg-red-700 text-white py-2 px-3 rounded-lg text-sm font-semibold transition-colors"
                  >
                    <Phone className="w-4 h-4" />
                    <span>Call Now</span>
                  </button>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}