import re
import time
from typing import Any, Dict, List, Callable

class Home:
    """
    smart home simulation class that manages device states and processes commands
    """
    def __init__(self, home_name="Mi Casa"):
        # initialize the event log and home state
        self.event_log = []
        self.state = {
            "nombre": home_name,
            "habitaciones": {
                "salon": {
                    "television": {
                        "estado": "apagado"
                    },
                    "termostato": {
                        "estado": "apagado", 
                        "temperatura_celsius": 22
                        }
                },
                "cocina": {
                    "luz": {
                        "estado": "apagado"
                    },
                    "horno": {
                        "estado": "apagado", 
                        "temperatura_celsius": "0"
                    },
                    "nevera": {
                        "estado": "encendido", 
                        "temperatura_celsius": "4"
                    }
                },
            }
        }
        # precompile command patterns
        self.command_patterns = self._build_command_patterns()

    def log_event(self, category: str, action: str, details: str) -> None:
        """log an event with timestamp"""
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        self.event_log.append({
            "timestamp": timestamp,
            "category": category,
            "action": action,
            "details": details
        })

    def update_device(self, room: str, device: str, property_name: str, value: Any) -> bool:
        """update a device property in a room"""
        if room not in self.state["habitaciones"]:
            self.log_event("system", "error", f"Room '{room}' not found")
            return False
        if device not in self.state["habitaciones"][room]:
            self.log_event("system", "error", f"Device '{device}' not found in '{room}'")
            return False
        if property_name not in self.state["habitaciones"][room][device]:
            self.log_event("system", "error", f"Property '{property_name}' not found for device '{device}'")
            return False
        
        old_value = self.state["habitaciones"][room][device][property_name]
        self.state["habitaciones"][room][device][property_name] = value
        self.log_event("device", "updated",
                       f"Updated {room}/{device}/{property_name} from '{old_value}' to '{value}'")
        return True

    def process_request(self, user_request: str, assistant_response: str = None) -> Dict[str, Any]:
        """process a user request and detect commands"""
        changes_made = {"success": False, "changes": [], "is_command": False}
        text_to_analyze = user_request.lower()
        if assistant_response and assistant_response != user_request:
            text_to_analyze += " " + assistant_response.lower()

        # first segmentation by punctuation
        segments = re.split(r'[.!?]', text_to_analyze)
        candidate_segments = []

        # check each segment for command keywords
        command_keywords = r"\b(?:enciende|apaga|activa|desactiva|abre|cierra|ajusta|pon)\b"
        for seg in segments:
            seg = seg.strip()
            if seg and re.search(command_keywords, seg):
                candidate_segments.append(seg)

        # if no candidate found, use n-gram sliding window
        if not candidate_segments:
            tokens = text_to_analyze.split()
            window_size = 3
            for i in range(len(tokens) - window_size + 1):
                window = " ".join(tokens[i:i+window_size])
                if re.search(command_keywords, window):
                    candidate_segments.append(window)

        # process each candidate segment with regex patterns
        if candidate_segments:
            changes_made["is_command"] = True
            for segment in candidate_segments:
                for cmd in self.command_patterns:
                    for match in cmd["regex"].finditer(segment):
                        try:
                            result = cmd["action"](match)
                            changes_made["changes"].append({
                                "command": cmd["name"],
                                "matched_text": match.group(0),
                                "success": bool(result)
                            })
                        except Exception as e:
                            changes_made["changes"].append({
                                "command": cmd["name"],
                                "matched_text": match.group(0),
                                "success": False,
                                "reason": str(e)
                            })

        if changes_made["changes"]:
            changes_made["success"] = any(change["success"] for change in changes_made["changes"])
        return changes_made

    def _build_command_patterns(self) -> List[Dict[str, Any]]:
        """build regex patterns for command recognition"""
        # courtesy prefix patterns
        prefix = r"(?:.*?(?:por\s+favor|oye|escucha|necesito\s+que|quiero\s+que|puedes|podrías|podrias|puedes|te\s+importaría|te\s+importaria|te\s+pido\s+que|me\s+gustaría\s+que|me\s+gustaria\s+que|hay\s+que|deberías|deberias|quisiera\s+que|me\s+ayudas\s+a|ayúdame\s+a|ayudame\s+a|haz\s+el\s+favor\s+de|sería\s+bueno|seria\s+bueno|¿puedes|me\s+haces\s+el\s+favor\s+de))?\s*"
        
        # action verbs
        turnon_verbs = r"(?:enciend(?:e|a)|activ(?:a|ar)|prend(?:e|a|er)|ilumin(?:a|ar)|conect(?:a|ar)|p[óo]n|poner|da(?:r|me)?|inicia(?:r)?|pon(?:er|ga|me|nos)?|activ(?:a|ar)|arranca(?:r)?)"
        turnoff_verbs = r"(?:apag(?:a|ar)|desactiv(?:a|ar)|desconect(?:a|ar)|quit(?:a|ar)|par(?:a|ar)|apag(?:a|ar)|desconect(?:a|ar)|interrump(?:e|ir)|deten(?:er)?|corta(?:r)?)"
        adjust_verbs = r"(?:ajust(?:a|ar)|configur(?:a|ar)|fij(?:a|ar)|coloc(?:a|ar)|establec(?:e|er)|cambi(?:a|ar)|pon(?:er|ga)?|modific(?:a|ar)|regula(?:r)?)"
        
        # optional articles and connectors
        articulo_opcional = r"(?:(?:el|la|los|las|un|una|unos|unas)\s+)?"
        conectores_habitacion = r"(?:(?:de|en|para|del|de\s+la|de\s+el)\s+)?"
        
        patterns = [
            # Lights On
            {
                "name": "light_on",
                "regex": re.compile(prefix + turnon_verbs + r"\s+" + articulo_opcional + r"(?:luz|luces|lámpara(?:s)?|bombilla(?:s)?|iluminación)(?:\s+" + conectores_habitacion + r"(?P<room>\w+))?",
                                    re.IGNORECASE),
                "action": lambda m: self._try_update_light(m.group("room") if m.group("room") else "salon", "encendido")
            },
            # Lights Off
            {
                "name": "light_off",
                "regex": re.compile(prefix + turnoff_verbs + r"\s+" + articulo_opcional + r"(?:luz|luces|lámpara(?:s)?|bombilla(?:s)?|iluminación)(?:\s+" + conectores_habitacion + r"(?P<room>\w+))?",
                                    re.IGNORECASE),
                "action": lambda m: self._try_update_light(m.group("room") if m.group("room") else "salon", "apagado")
            },
            # TV On (assumed to be in salon)
            {
                "name": "tv_on",
                "regex": re.compile(prefix + turnon_verbs + r"\s+" + articulo_opcional + r"(?:televisión|televisor|tele(?:visión)?|tv|plasma|pantalla)(?:\s+" + conectores_habitacion + r"(?P<room>salon|sala))?",
                                    re.IGNORECASE),
                "action": lambda m: self.update_device("salon", "television", "estado", "encendido")
            },
            # TV Off
            {
                "name": "tv_off",
                "regex": re.compile(prefix + turnoff_verbs + r"\s+" + articulo_opcional + r"(?:televisión|televisor|tele(?:visión)?|tv|plasma|pantalla)(?:\s+" + conectores_habitacion + r"(?P<room>salon|sala))?",
                                    re.IGNORECASE),
                "action": lambda m: self.update_device("salon", "television", "estado", "apagado")
            },
            # Set Temperature (Thermostat/Air)
            {
                "name": "temp_set",
                "regex": re.compile(prefix + adjust_verbs + r"\s+" + articulo_opcional + r"(?:termostato|temperatura|aire(?:\s+acondicionado)?)(?:\s+" + conectores_habitacion + r"(?P<room>\w+))?\s+(?:a|en|para|como)?\s*(?P<temp>\d+)(?:\s*(?:grados|°|°C|ºC|C))?",
                                    re.IGNORECASE),
                "action": lambda m: self._try_update_temperature(m.group("room") if m.group("room") else "salon", m.group("temp"))
            },
            # Climate Device On
            {
                "name": "climate_on",
                "regex": re.compile(prefix + turnon_verbs + r"\s+" + articulo_opcional + r"(?:termostato|aire(?:\s+acondicionado)?|climatización|climatizador|calefacción|calefactor)(?:\s+" + conectores_habitacion + r"(?P<room>\w+))?",
                                    re.IGNORECASE),
                "action": lambda m: self._try_update_climate_device(m.group("room") if m.group("room") else "salon", "encendido")
            },
            # Climate Device Off
            {
                "name": "climate_off",
                "regex": re.compile(prefix + turnoff_verbs + r"\s+" + articulo_opcional + r"(?:termostato|aire(?:\s+acondicionado)?|climatización|climatizador|calefacción|calefactor)(?:\s+" + conectores_habitacion + r"(?P<room>\w+))?",
                                    re.IGNORECASE),
                "action": lambda m: self._try_update_climate_device(m.group("room") if m.group("room") else "salon", "apagado")
            },
            # Oven On (always in cocina)
            {
                "name": "oven_on",
                "regex": re.compile(prefix + turnon_verbs + r"\s+" + articulo_opcional + r"(?:horno|estufa)(?:\s+" + conectores_habitacion + r"(?P<room>cocina))?",
                                    re.IGNORECASE),
                "action": lambda m: self.update_device("cocina", "horno", "estado", "encendido")
            },
            # Oven Off
            {
                "name": "oven_off",
                "regex": re.compile(prefix + turnoff_verbs + r"\s+" + articulo_opcional + r"(?:horno|estufa)(?:\s+" + conectores_habitacion + r"(?P<room>cocina))?",
                                    re.IGNORECASE),
                "action": lambda m: self.update_device("cocina", "horno", "estado", "apagado")
            },
            # Oven Temperature Set
            {
                "name": "oven_temp",
                "regex": re.compile(prefix + adjust_verbs + r"\s+" + articulo_opcional + r"(?:horno|temperatura\s+del\s+horno|calor\s+del\s+horno)(?:\s+" + conectores_habitacion + r"(?P<room>cocina))?\s+(?:a|en|para|como)?\s*(?P<temp>\d+)(?:\s*(?:grados|°|°C|ºC|C))?",
                                    re.IGNORECASE),
                "action": lambda m: self.update_device("cocina", "horno", "temperatura_celsius", m.group("temp"))
            },
        ]
        return patterns

    def _try_update_light(self, room: str, state: str) -> bool:
        """update light state in a room using room aliases"""
        room_mapping = {
            "sala": "salon",
            "salón": "salon",
            "living": "salon"
        }
        normalized_room = room_mapping.get(room.lower(), room.lower())
        if normalized_room in self.state["habitaciones"] and "luz" in self.state["habitaciones"][normalized_room]:
            return self.update_device(normalized_room, "luz", "estado", state)
        return False

    def _try_update_temperature(self, room: str, temperature: str) -> bool:
        """update temperature on the appropriate climate control device"""
        room_mapping = {
            "sala": "salon",
            "salón": "salon",
            "living": "salon"
        }
        climate_devices = {
            "salon": "termostato",
            "cocina": None  # cocina doesn't have a dedicated climate control
        }
        normalized_room = room_mapping.get(room.lower(), room.lower())
        device = climate_devices.get(normalized_room)
        if not device:
            return False
        self.update_device(normalized_room, device, "estado", "encendido")
        return self.update_device(normalized_room, device, "temperatura_celsius", temperature)

    def _try_update_climate_device(self, room: str, state: str) -> bool:
        """update on/off state for a climate control device"""
        room_mapping = {
            "sala": "salon",
            "salón": "salon",
            "living": "salon"
        }
        climate_devices = {
            "salon": "termostato",
            "cocina": None
        }
        normalized_room = room_mapping.get(room.lower(), room.lower())
        device = climate_devices.get(normalized_room)
        if not device:
            return False
        return self.update_device(normalized_room, device, "estado", state)