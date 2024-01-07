package data

import (
	"encoding/json"
	"io"
	"log"
	"os"
)

const DATA_FILE = "data/vanilla-1.1.19-mod.json"

type Item struct {
	Name          string
	Group         string
	IconCol       int
	IconRow       int
	LocalizedName map[string]string
	Effect        Effect
	Order         string
	StackSize     int
	Subgroup      string
	Type          string
}

type Effect struct {
	Consumption  float64
	Speed        float64
	Productivity float64
	Pollution    float64
}

type Recipe struct {
	Category       string
	Enabled        bool
	EnergyRequired float64
	IconCol        int
	IconRow        int
	Ingredients    []struct {
		Amount float64
		Name   string
		Type   string
	}
	LocalizedName map[string]string
	Name          string
	Order         string
	Results       []struct {
		Amount float64
		Name   string
		Type   string
	}
	Subgroup string
	Type     string
}

type Resource struct {
	IconCol       int
	IconRow       int
	LocalizedName map[string]string
	Category      string
	Mineable      Mineable
	Name          string
}

type Mineable struct {
	MiningParticle string
	MiningTime     float64
	Results        []struct {
		Amount float64
		Name   string
		Type   string
	}
}

type AssemblingMachine struct {
	CraftingCategories []string
	AllowedEffects     []string
	CraftingSpeed      float64
	EnergyUsage        float64
	IconCol            int
	IconRow            int
	LocalizedName      map[string]string
	ModuleSlots        int
	Name               string
}

type EnergySource struct {
	Effectivity        float64 // Optional, not present in all drills
	EmissionsPerMinute float64
	FuelCategory       string // Optional, not present in all drills
	FuelInventorySize  int    // Optional, not present in all drills
	Type               string
	UsagePriority      string // Optional, not present in all drills
}

type LightFlicker struct {
	Color [3]int
}

type Smoke struct {
	Deviation [2]float64
	Frequency float64
	Name      string
}

type MiningDrill struct {
	EnergySource       EnergySource
	EnergyUsage        float64
	IconCol            int
	IconRow            int
	LocalizedName      map[string]string
	MiningSpeed        float64
	ModuleSlots        int
	Name               string
	ResourceCategories []string
}

type Furnace struct {
	AllowedEffects     []string          `json:"allowed_effects,omitempty"`
	CraftingCategories []string          `json:"crafting_categories"`
	CraftingSpeed      float64           `json:"crafting_speed"`
	EnergySource       EnergySource      `json:"energy_source"`
	EnergyUsage        int               `json:"energy_usage"`
	IconCol            int               `json:"icon_col"`
	IconRow            int               `json:"icon_row"`
	LocalizedName      map[string]string `json:"localized_name"`
	ModuleSlots        int               `json:"module_slots"`
	Name               string            `json:"name"`
}

type Module struct {
	Name string
}

type Schema struct {
	Items              map[string]Item
	Recipes            map[string]Recipe
	Resources          map[string]Resource
	AssemblingMachines map[string]AssemblingMachine
	MiningDrills       map[string]MiningDrill
	Furnaces           map[string]Furnace
	Modules            map[string]Module
}

var Data Schema

func load_data(path string) (data map[string]interface{}) {
	data_file, err := os.Open(path)
	if err != nil {
		log.Fatalf("Error: %e", err)
	}
	defer data_file.Close()

	bytes, err := io.ReadAll(data_file)
	if err != nil {
		log.Fatalf("Error: %e", err)
	}

	err = json.Unmarshal(bytes, &data)
	if err != nil {
		log.Fatalf("Error: %e", err)
	}

	return data
}

func (schema *Schema) load_item_data(data map[string]interface{}) {
	if schema.Items == nil {
		schema.Items = make(map[string]Item)
	}

	if items, ok := data["items"].(map[string]interface{}); ok {
		for key, item_data := range items {
			item_data, ok := item_data.(map[string]interface{})
			if !ok {
				log.Fatalf("Received invalid data when parsing item data")
			}

			var item Item
			item.Name = item_data["name"].(string)
			item.Group = item_data["group"].(string)
			item.IconCol = int(item_data["icon_col"].(float64))
			item.IconRow = int(item_data["icon_row"].(float64))
			item.Subgroup = item_data["subgroup"].(string)
			item.Type = item_data["type"].(string)

			if order, ok := item_data["order"].(string); ok {
				item.Order = order
			} else {
				item.Order = ""
			}

			item.LocalizedName = make(map[string]string)
			if localized_name, ok := item_data["localized_name"].(map[string]interface{}); ok {
				for k, v := range localized_name {
					item.LocalizedName[k] = v.(string)
				}
			}

			if stack_size, ok := item_data["stack_size"].(float64); ok {
				item.StackSize = int(stack_size)
			} else {
				item.StackSize = 0
			}

			if effect_data, ok := item_data["effect"].(map[string]interface{}); ok {
				for effect, bonus := range effect_data {
					bonus := (bonus.(map[string]interface{}))["bonus"].(float64)

					switch effect {
					case "consumption":
						item.Effect.Consumption = bonus
					case "pollution":
						item.Effect.Pollution = bonus
					case "productivity":
						item.Effect.Productivity = bonus
					case "speed":
						item.Effect.Speed = bonus
					}
				}
			}

			schema.Items[key] = item
		}
	} else {
		log.Fatalf("Received invalid data when parsing item data")
	}
}

func (schema *Schema) load_recipe_data(data map[string]interface{}) {
	if schema.Recipes == nil {
		schema.Recipes = make(map[string]Recipe)
	}

	if recipes, ok := data["recipes"].(map[string]interface{}); ok {
		for key, recipe_data := range recipes {
			recipe_data, ok := recipe_data.(map[string]interface{})
			if !ok {
				log.Fatalf("Received invalid recipe data: %v", recipe_data)
			}

			var recipe Recipe
			recipe.Category = recipe_data["category"].(string)
			recipe.EnergyRequired = recipe_data["energy_required"].(float64)
			recipe.IconCol = int(recipe_data["icon_col"].(float64))
			recipe.IconRow = int(recipe_data["icon_row"].(float64))
			recipe.Name = recipe_data["name"].(string)
			recipe.Subgroup = recipe_data["subgroup"].(string)
			recipe.Type = recipe_data["type"].(string)

			if enabled, ok := recipe_data["enabled"].(bool); ok {
				recipe.Enabled = enabled
			} else {
				recipe.Enabled = true
			}

			if order, ok := recipe_data["order"].(string); ok {
				recipe.Order = order
			} else {
				recipe.Order = ""
			}

			recipe.LocalizedName = make(map[string]string)
			if localized_name, ok := recipe_data["localized_name"].(map[string]interface{}); ok {
				for k, v := range localized_name {
					recipe.LocalizedName[k] = v.(string)
				}
			}

			ingredients := recipe_data["ingredients"].([]interface{})
			recipe.Ingredients = make([]struct {
				Amount float64
				Name   string
				Type   string
			}, 0, len(ingredients))
			for _, ingredient_data := range ingredients {
				ingredient_data := ingredient_data.(map[string]interface{})

				var ingredient struct {
					Amount float64
					Name   string
					Type   string
				}
				ingredient.Amount = ingredient_data["amount"].(float64)
				ingredient.Name = ingredient_data["name"].(string)

				item_type, ok := ingredient_data["type"].(string)
				if ok {
					ingredient.Type = item_type
				} else {
					ingredient.Type = "solid"
				}

				recipe.Ingredients = append(recipe.Ingredients, ingredient)
			}

			results := recipe_data["results"].([]interface{})
			recipe.Results = make([]struct {
				Amount float64
				Name   string
				Type   string
			}, 0, len(results))
			for _, result_data := range results {
				result_data := result_data.(map[string]interface{})

				var result struct {
					Amount float64
					Name   string
					Type   string
				}
				result.Amount = result_data["amount"].(float64)
				result.Name = result_data["name"].(string)

				item_type, ok := result_data["type"].(string)
				if ok {
					result.Type = item_type
				} else {
					result.Type = "solid"
				}

				recipe.Results = append(recipe.Results, result)
			}

			schema.Recipes[key] = recipe
		}
	}
}

func (schema *Schema) load_resource_data(data map[string]interface{}) {
	if schema.Resources == nil {
		schema.Resources = make(map[string]Resource)
	}

	if resources, ok := data["resource"].(map[string]interface{}); ok {
		for key, resource_data := range resources {
			resource_data, ok := resource_data.(map[string]interface{})
			if !ok {
				log.Fatalf("Received invalid resource data: %v", resource_data)
			}

			var resource Resource
			resource.IconCol = int(resource_data["icon_col"].(float64))
			resource.IconRow = int(resource_data["icon_row"].(float64))
			resource.Name = resource_data["name"].(string)

			if category, ok := resource_data["category"].(string); ok {
				resource.Category = category
			} else {
				resource.Category = "basic-solid"
			}

			resource.LocalizedName = make(map[string]string)
			for k, v := range resource_data["localized_name"].(map[string]interface{}) {
				resource.LocalizedName[k] = v.(string)
			}

			mineable_data := resource_data["minable"].(map[string]interface{})

			mining_particle, ok := mineable_data["mining_particle"].(string)
			if ok {
				resource.Mineable.MiningParticle = mining_particle
			} else {
				resource.Mineable.MiningParticle = ""
			}

			resource.Mineable.MiningTime = mineable_data["mining_time"].(float64)

			results := mineable_data["results"].([]interface{})

			resource.Mineable.Results = make([]struct {
				Amount float64
				Name   string
				Type   string
			}, 0, len(results))
			for _, result_data := range results {
				result_data := result_data.(map[string]interface{})

				var result struct {
					Amount float64
					Name   string
					Type   string
				}

				if amount, ok := result_data["amount"].(float64); ok {
					result.Amount = amount
				} else if amount, ok := result_data["amount_min"].(float64); ok {
					result.Amount = amount
				}

				result.Name = result_data["name"].(string)

				if result_type, ok := result_data["type"].(string); ok {
					result.Type = result_type
				} else {
					result.Type = "solid"
				}

				resource.Mineable.Results = append(resource.Mineable.Results, result)
			}

			schema.Resources[key] = resource
		}
	}
}

func (schema *Schema) load_assembling_machine_data(data map[string]interface{}) {
	if schema.AssemblingMachines == nil {
		schema.AssemblingMachines = make(map[string]AssemblingMachine)
	}

	if machines, ok := data["assembling-machine"].(map[string]interface{}); ok {
		for key, machineData := range machines {
			machineDataMap, ok := machineData.(map[string]interface{})
			if !ok {
				log.Fatalf("Received invalid assembling machine data: %v", machineData)
			}

			var machine AssemblingMachine
			machine.CraftingSpeed = machineDataMap["crafting_speed"].(float64)
			machine.EnergyUsage = machineDataMap["energy_usage"].(float64)
			machine.IconCol = int(machineDataMap["icon_col"].(float64))
			machine.IconRow = int(machineDataMap["icon_row"].(float64))
			machine.Name = machineDataMap["name"].(string)
			machine.ModuleSlots = int(machineDataMap["module_slots"].(float64))

			machine.LocalizedName = make(map[string]string)
			if localized_name, ok := machineDataMap["localized_name"].(map[string]interface{}); ok {
				for k, v := range localized_name {
					machine.LocalizedName[k] = v.(string)
				}
			}

			craftingCategories := machineDataMap["crafting_categories"].([]interface{})
			for _, category := range craftingCategories {
				machine.CraftingCategories = append(machine.CraftingCategories, category.(string))
			}

			if allowedEffects, ok := machineDataMap["allowed_effects"].([]interface{}); ok {
				for _, effect := range allowedEffects {
					machine.AllowedEffects = append(machine.AllowedEffects, effect.(string))
				}
			}

			schema.AssemblingMachines[key] = machine
		}
	}
}

func (schema *Schema) load_mining_drill_data(data map[string]interface{}) {
	if schema.MiningDrills == nil {
		schema.MiningDrills = make(map[string]MiningDrill)
	}

	if drills, ok := data["mining-drill"].(map[string]interface{}); ok {
		for key, drillData := range drills {
			drillDataMap, ok := drillData.(map[string]interface{})
			if !ok {
				log.Fatalf("Received invalid mining drill data: %v", drillData)
			}

			var drill MiningDrill
			// Energy source parsing
			if esData, ok := drillDataMap["energy_source"].(map[string]interface{}); ok {
				var es EnergySource
				es.EmissionsPerMinute = esData["emissions_per_minute"].(float64)
				es.Type = esData["type"].(string)

				// Optional fields in energy source
				if val, ok := esData["effectivity"].(float64); ok {
					es.Effectivity = val
				}
				if val, ok := esData["fuel_category"].(string); ok {
					es.FuelCategory = val
				}
				if val, ok := esData["fuel_inventory_size"].(float64); ok {
					es.FuelInventorySize = int(val)
				}
				if val, ok := esData["usage_priority"].(string); ok {
					es.UsagePriority = val
				}

				drill.EnergySource = es
			}

			if energyUsage, ok := drillDataMap["energy_usage"].(float64); ok {
				drill.EnergyUsage = energyUsage
			}
			drill.IconCol = int(drillDataMap["icon_col"].(float64))
			drill.IconRow = int(drillDataMap["icon_row"].(float64))
			drill.Name = drillDataMap["name"].(string)
			drill.MiningSpeed = drillDataMap["mining_speed"].(float64)
			drill.ModuleSlots = int(drillDataMap["module_slots"].(float64))

			drill.LocalizedName = make(map[string]string)
			if localized_name, ok := drillDataMap["localized_name"].(map[string]interface{}); ok {
				for k, v := range localized_name {
					drill.LocalizedName[k] = v.(string)
				}
			}

			resourceCategories := drillDataMap["resource_categories"].([]interface{})
			for _, category := range resourceCategories {
				drill.ResourceCategories = append(drill.ResourceCategories, category.(string))
			}

			schema.MiningDrills[key] = drill
		}
	}
}

func (schema *Schema) load_furnace_data(data map[string]interface{}) {
	if schema.Furnaces == nil {
		schema.Furnaces = make(map[string]Furnace)
	}

	if furnaces, ok := data["furnace"].(map[string]interface{}); ok {
		for key, furnaceData := range furnaces {
			furnaceMap, ok := furnaceData.(map[string]interface{})
			if !ok {
				log.Fatalf("Received invalid furnace data: %v", furnaceData)
			}

			var furnace Furnace
			if allowedEffects, ok := furnaceMap["allowed_effects"].([]interface{}); ok {
				for _, effect := range allowedEffects {
					furnace.AllowedEffects = append(furnace.AllowedEffects, effect.(string))
				}
			}

			if craftingCategories, ok := furnaceMap["crafting_categories"].([]interface{}); ok {
				for _, category := range craftingCategories {
					furnace.CraftingCategories = append(furnace.CraftingCategories, category.(string))
				}
			}

			if craftingSpeed, ok := furnaceMap["crafting_speed"].(float64); ok {
				furnace.CraftingSpeed = craftingSpeed
			}

			if energyUsage, ok := furnaceMap["energy_usage"].(float64); ok {
				furnace.EnergyUsage = int(energyUsage)
			}

			if iconCol, ok := furnaceMap["icon_col"].(float64); ok {
				furnace.IconCol = int(iconCol)
			}

			if iconRow, ok := furnaceMap["icon_row"].(float64); ok {
				furnace.IconRow = int(iconRow)
			}

			if localizedName, ok := furnaceMap["localized_name"].(map[string]interface{}); ok {
				furnace.LocalizedName = make(map[string]string)
				for lang, name := range localizedName {
					furnace.LocalizedName[lang] = name.(string)
				}
			}

			if moduleSlots, ok := furnaceMap["module_slots"].(float64); ok {
				furnace.ModuleSlots = int(moduleSlots)
			}

			if name, ok := furnaceMap["name"].(string); ok {
				furnace.Name = name
			}

			// Parsing nested EnergySource
			if esData, ok := furnaceMap["energy_source"].(map[string]interface{}); ok {
				var es EnergySource
				if emissions, ok := esData["emissions_per_minute"].(float64); ok {
					es.EmissionsPerMinute = emissions
				}
				if esType, ok := esData["type"].(string); ok {
					es.Type = esType
				}
				// ... Continue parsing other fields of EnergySource ...
				furnace.EnergySource = es
			}

			schema.Furnaces[key] = furnace
		}
	} else {
		log.Fatalf("Received invalid data when parsing furnace data")
	}
}

func (schema *Schema) load_module_data(data map[string]interface{}) {
	if schema.Modules == nil {
		schema.Modules = make(map[string]Module)
	}

	for _, moduleName := range data["modules"].([]interface{}) {
		moduleName := moduleName.(string)
		schema.Modules[moduleName] = Module{
			Name: moduleName,
		}
	}
}

func init() {
	data := load_data(DATA_FILE)

	Data.load_item_data(data)
	Data.load_recipe_data(data)
	Data.load_resource_data(data)
	Data.load_assembling_machine_data(data)
	Data.load_mining_drill_data(data)
	Data.load_furnace_data(data)
	Data.load_module_data(data)
}
