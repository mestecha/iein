import re
import time
from typing import Any, Dict, List, Callable

class Home:
    """
    Smart Home Simulation Class

    This class simulates a smart home environment with multiple rooms, devices, 
    security features, and environmental controls. It provides methods to interact 
    with and modify the smart home state.
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
        # "seguridad": {
        #     "puerta_principal": {"estado": "cerrado", "ultima_vez_accedido": None},
        #     "puerta_garaje": {"estado": "cerrado"},
        #     "alarma": {"estado": "encendido"}
        # }
        # "dormitorio": {
        #     "luz": {"estado": "apagado"},
        #     "aire": {"estado": "apagado", "temperatura_celsius": "0"}
        # }
        # Precompile command patterns
        self.command_patterns = self._build_command_patterns()

    def log_event(self, category: str, action: str, details: str) -> None:
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        self.event_log.append({
            "timestamp": timestamp,
            "category": category,
            "action": action,
            "details": details
        })

    def update_device(self, room: str, device: str, property_name: str, value: Any) -> bool:
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

    # def update_security(self, security_item: str, property_name: str, value: Any) -> bool:
    #     if security_item not in self.state["seguridad"]:
    #         self.log_event("system", "error", f"Security item '{security_item}' not found")
    #         return False
    #     if property_name not in self.state["seguridad"][security_item]:
    #         self.log_event("system", "error", f"Property '{property_name}' not found for '{security_item}'")
    #         return False
        
    #     if security_item == "puerta_principal" and property_name == "estado" and value in ["abierto", "desbloqueado"]:
    #         self.state["seguridad"][security_item]["ultima_vez_accedido"] = time.strftime("%Y-%m-%d %H:%M:%S")
        
    #     old_value = self.state["seguridad"][security_item][property_name]
    #     self.state["seguridad"][security_item][property_name] = value
    #     self.log_event("security", "updated",
    #                    f"Updated seguridad/{security_item}/{property_name} from '{old_value}' to '{value}'")
    #     return True

    def process_request(self, user_request: str, assistant_response: str = None) -> Dict[str, Any]:
        changes_made = {"success": False, "changes": [], "is_command": False}
        text_to_analyze = user_request.lower()
        if assistant_response and assistant_response != user_request:
            text_to_analyze += " " + assistant_response.lower()

        # First segmentation by punctuation.
        segments = re.split(r'[.!?]', text_to_analyze)
        candidate_segments = []

        # Check each segment for command keywords.
        command_keywords = r"\b(?:enciende|apaga|activa|desactiva|abre|cierra|ajusta|pon)\b"
        for seg in segments:
            seg = seg.strip()
            if seg and re.search(command_keywords, seg):
                candidate_segments.append(seg)

        # If no candidate found, use n-gram sliding window (e.g., 3-word chunks).
        if not candidate_segments:
            tokens = text_to_analyze.split()
            window_size = 3
            for i in range(len(tokens) - window_size + 1):
                window = " ".join(tokens[i:i+window_size])
                if re.search(command_keywords, window):
                    candidate_segments.append(window)

        # process each candidate segment with your regex patterns.
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
        """
        Build a list of command patterns. Each command is represented as a dict with:
          - name: identifier for the command
          - regex: a compiled regex pattern (using named groups)
          - action: a lambda to execute when the pattern is matched
        """
        # Prefijo expandido con más frases de cortesía y construcciones
        prefix = r"(?:.*?(?:por\s+favor|oye|escucha|necesito\s+que|quiero\s+que|puedes|podrías|podrias|puedes|te\s+importaría|te\s+importaria|te\s+pido\s+que|me\s+gustaría\s+que|me\s+gustaria\s+que|hay\s+que|deberías|deberias|quisiera\s+que|me\s+ayudas\s+a|ayúdame\s+a|ayudame\s+a|haz\s+el\s+favor\s+de|sería\s+bueno|seria\s+bueno|¿puedes|me\s+haces\s+el\s+favor\s+de))?\s*"
        
        # Verbos de acción expandidos
        turnon_verbs = r"(?:enciend(?:e|a)|activ(?:a|ar)|prend(?:e|a|er)|ilumin(?:a|ar)|conect(?:a|ar)|p[óo]n|poner|da(?:r|me)?|inicia(?:r)?|pon(?:er|ga|me|nos)?|activ(?:a|ar)|arranca(?:r)?)"
        turnoff_verbs = r"(?:apag(?:a|ar)|desactiv(?:a|ar)|desconect(?:a|ar)|quit(?:a|ar)|par(?:a|ar)|apag(?:a|ar)|desconect(?:a|ar)|interrump(?:e|ir)|deten(?:er)?|corta(?:r)?)"
        adjust_verbs = r"(?:ajust(?:a|ar)|configur(?:a|ar)|fij(?:a|ar)|coloc(?:a|ar)|establec(?:e|er)|cambi(?:a|ar)|pon(?:er|ga)?|modific(?:a|ar)|regula(?:r)?)"
        open_verbs = r"(?:abr(?:e|ir)|desbloque(?:a|ar)|desatranc(?:a|ar)|libera(?:r)?)"
        close_verbs = r"(?:cierr(?:a|e)|bloque(?:a|ar)|atranc(?:a|ar)|asegur(?:a|ar)|pon\s+seguro\s+a)"
        
        # Artículos flexibles
        articulo_opcional = r"(?:(?:el|la|los|las|un|una|unos|unas)\s+)?"
        
        # Conectores para habitaciones
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
            # # Security - Main Door Open
            # {
            #     "name": "door_main_open",
            #     "regex": re.compile(prefix + r"(?:abre|desbloquea|desatranca)\s+(?:la\s+)?puerta\s+(?:principal|de\s+entrada|del\s+frente)",
            #                         re.IGNORECASE),
            #     "action": lambda m: self.update_security("puerta_principal", "estado", "abierto")
            # },
            # # Security - Main Door Close
            # {
            #     "name": "door_main_close",
            #     "regex": re.compile(prefix + r"(?:cierra|bloquea|atranca|pon\s+seguro\s+a)\s+(?:la\s+)?puerta\s+(?:principal|de\s+entrada|del\s+frente)",
            #                         re.IGNORECASE),
            #     "action": lambda m: self.update_security("puerta_principal", "estado", "cerrado")
            # },
            # # Security - Garage Door Open
            # {
            #     "name": "door_garage_open",
            #     "regex": re.compile(prefix + open_verbs + r"\s+" + articulo_opcional + r"(?:puerta\s+(?:del\s+|de\s+la\s+)?(?:garaje|cochera|estacionamiento|garage)|(?:garaje|cochera|estacionamiento|garage))",
            #                         re.IGNORECASE),
            #     "action": lambda m: self.update_security("puerta_garaje", "estado", "abierto")
            # },
            # # Security - Garage Door Close
            # {
            #     "name": "door_garage_close",
            #     "regex": re.compile(prefix + close_verbs + r"\s+" + articulo_opcional + r"(?:puerta\s+(?:del\s+|de\s+la\s+)?(?:garaje|cochera|estacionamiento|garage)|(?:garaje|cochera|estacionamiento|garage))",
            #                         re.IGNORECASE),
            #     "action": lambda m: self.update_security("puerta_garaje", "estado", "cerrado")
            # },
            # # Security - Alarm On
            # {
            #     "name": "alarm_on",
            #     "regex": re.compile(prefix + turnon_verbs + r"\s+" + articulo_opcional + r"(?:alarma|sistema\s+(?:de\s+)?(?:seguridad|alarma)|seguridad|vigilancia)",
            #                         re.IGNORECASE),
            #     "action": lambda m: self.update_security("alarma", "estado", "encendido")
            # },
            # # Security - Alarm Off
            # {
            #     "name": "alarm_off",
            #     "regex": re.compile(prefix + turnoff_verbs + r"\s+" + articulo_opcional + r"(?:alarma|sistema\s+(?:de\s+)?(?:seguridad|alarma)|seguridad|vigilancia)",
            #                         re.IGNORECASE),
            #     "action": lambda m: self.update_security("alarma", "estado", "apagado")
            # }
        ]
        return patterns

    def _try_update_light(self, room: str, state: str) -> bool:
        """
        Update light in a room. Uses room aliases.
        """
        room_mapping = {
            "sala": "salon",
            "salón": "salon",
            "living": "salon",
            "habitación": "dormitorio",
            "cuarto": "dormitorio",
            "bedroom": "dormitorio",
        }
        normalized_room = room_mapping.get(room.lower(), room.lower())
        if normalized_room in self.state["habitaciones"] and "luz" in self.state["habitaciones"][normalized_room]:
            return self.update_device(normalized_room, "luz", "estado", state)
        return False

    def _try_update_temperature(self, room: str, temperature: str) -> bool:
        """
        Update temperature on the appropriate climate control device.
        """
        room_mapping = {
            "sala": "salon",
            "salón": "salon",
            "living": "salon",
            "habitación": "dormitorio",
            "cuarto": "dormitorio",
            "bedroom": "dormitorio",
        }
        climate_devices = {
            "salon": "termostato",
            "dormitorio": "aire",
            "cocina": None  # Cocina doesn't have a dedicated climate control
        }
        normalized_room = room_mapping.get(room.lower(), room.lower())
        device = climate_devices.get(normalized_room)
        if not device:
            return False
        self.update_device(normalized_room, device, "estado", "encendido")
        return self.update_device(normalized_room, device, "temperatura_celsius", temperature)

    def _try_update_climate_device(self, room: str, state: str) -> bool:
        """
        Update on/off state for a climate control device.
        """
        room_mapping = {
            "sala": "salon",
            "salón": "salon",
            "living": "salon",
            "habitación": "dormitorio",
            "cuarto": "dormitorio",
            "bedroom": "dormitorio",
        }
        climate_devices = {
            "salon": "termostato",
            "dormitorio": "aire",
            "cocina": None
        }
        normalized_room = room_mapping.get(room.lower(), room.lower())
        device = climate_devices.get(normalized_room)
        if not device:
            return False
        return self.update_device(normalized_room, device, "estado", state)