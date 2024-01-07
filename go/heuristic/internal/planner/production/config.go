package production

import (
	"errors"
	"fmt"
	"log"

	"github.com/factorio-ai/internal/data"
	"github.com/factorio-ai/internal/utils"
)

type AssemblingMachine struct {
	craftingCategories []string
	craftingSpeed      float64
	moduleSlots        int

	energySource utils.EnergySource

	reference interface{}
}

func NewAssemblingMachineFrom(key interface{}, am data.AssemblingMachine) (res AssemblingMachine) {
	res.craftingCategories = make([]string, len(am.CraftingCategories))
	copy(res.craftingCategories, am.CraftingCategories)

	res.craftingSpeed = am.CraftingSpeed
	res.moduleSlots = am.ModuleSlots
	res.reference = key

	res.energySource.EnergyType = utils.ELECTRIC
	res.energySource.EnergyAmount = am.EnergyUsage

	return res
}

type MiningDrill struct {
	miningSpeed        float64
	moduleSlots        int
	resourceCategories []string

	energySource utils.EnergySource

	reference interface{}
}

func NewMiningDrillFrom(key interface{}, md data.MiningDrill) (res MiningDrill) {
	res.resourceCategories = make([]string, len(md.ResourceCategories))
	copy(res.resourceCategories, md.ResourceCategories)

	res.miningSpeed = md.MiningSpeed
	res.moduleSlots = md.ModuleSlots
	res.reference = key

	switch md.EnergySource.Type {
	case "electric":
		res.energySource.EnergyType = utils.ELECTRIC
	default:
		res.energySource.EnergyType = utils.COMBUSTION
	}

	res.energySource.EnergyAmount = md.EnergyUsage

	return res
}

type Furnace struct {
	smeltingSpeed      float64
	moduleSlots        int
	craftingCategories []string

	energySource utils.EnergySource

	reference interface{}
}
type FurnaceID int

func NewFurnaceFrom(key interface{}, f data.Furnace) (res Furnace) {
	res.craftingCategories = make([]string, len(f.CraftingCategories))
	copy(res.craftingCategories, f.CraftingCategories)

	res.moduleSlots = f.ModuleSlots
	res.smeltingSpeed = f.CraftingSpeed

	switch f.EnergySource.Type {
	case "electric":
		res.energySource.EnergyType = utils.ELECTRIC
	default:
		res.energySource.EnergyType = utils.COMBUSTION
	}

	res.energySource.EnergyAmount = float64(f.EnergyUsage)

	res.reference = key

	return res
}

type Effect struct {
	consumption  float64
	productivity float64
	speed        float64
}

func NewEffect() Effect {
	return Effect{
		consumption:  1,
		productivity: 1,
		speed:        1,
	}
}

func (effect Effect) Value() float64 {
	return effect.productivity * effect.speed / effect.consumption
}

func (effect Effect) Decompose() (float64, float64, float64) {
	return effect.consumption, effect.productivity, effect.speed
}

type Module struct {
	effect Effect

	reference interface{}
}

func NewModuleFrom(key interface{}, modItem data.Item) (res Module) {
	res.effect.consumption = modItem.Effect.Consumption
	res.effect.productivity = modItem.Effect.Productivity
	res.effect.speed = modItem.Effect.Speed

	return res
}

type ConfigOptions struct {
	assemblingMachineOptions []AssemblingMachine
	miningDrillOptions       []MiningDrill
	furnaceOptions           []Furnace
	moduleOptions            []Module
}

type AssemblingMachineID int
type MiningDrillID int
type ModuleID int

func NewConfigOptionsFrom(data data.Schema) (res ConfigOptions) {
	for key, am := range data.AssemblingMachines {
		res.assemblingMachineOptions = append(res.assemblingMachineOptions, NewAssemblingMachineFrom(key, am))
	}

	for key, md := range data.MiningDrills {
		res.miningDrillOptions = append(res.miningDrillOptions, NewMiningDrillFrom(key, md))
	}

	for key, f := range data.Furnaces {
		res.furnaceOptions = append(res.furnaceOptions, NewFurnaceFrom(key, f))
	}

	for key, mod := range data.Modules {
		modItem := data.Items[mod.Name]
		res.moduleOptions = append(res.moduleOptions, NewModuleFrom(key, modItem))
	}

	return res
}

func (configOptions *ConfigOptions) GetAssemblingMachineID(key interface{}) (AssemblingMachineID, error) {
	for i, am := range configOptions.assemblingMachineOptions {
		if am.reference == key {
			return AssemblingMachineID(i), nil
		}
	}
	return AssemblingMachineID(-1), fmt.Errorf("no assembly machine registered for: %v", key)
}

// GetMiningDrillID retrieves the ID of a Mining Drill based on the key.
func (configOptions *ConfigOptions) GetMiningDrillID(key interface{}) (MiningDrillID, error) {
	for i, md := range configOptions.miningDrillOptions {
		if md.reference == key {
			return MiningDrillID(i), nil
		}
	}
	return MiningDrillID(-1), fmt.Errorf("no mining drill registered for: %v", key)
}

// GetModuleID retrieves the ID of a Module based on the key.
func (configOptions *ConfigOptions) GetModuleID(key interface{}) (ModuleID, error) {
	for i, mod := range configOptions.moduleOptions {
		if mod.reference == key {
			return ModuleID(i), nil
		}
	}
	return ModuleID(-1), fmt.Errorf("no module registered for: %v", key)
}

// GetAssemblingMachineFromID retrieves an AssemblingMachine instance by its ID.
func (configOptions *ConfigOptions) GetAssemblingMachineFromID(id AssemblingMachineID) (AssemblingMachine, error) {
	if id >= 0 && int(id) < len(configOptions.assemblingMachineOptions) {
		return configOptions.assemblingMachineOptions[id], nil
	}
	return AssemblingMachine{}, fmt.Errorf("invalid AssemblingMachineID: %d", id)
}

// GetMiningDrillFromID retrieves a MiningDrill instance by its ID.
func (configOptions *ConfigOptions) GetMiningDrillFromID(id MiningDrillID) (MiningDrill, error) {
	if id >= 0 && int(id) < len(configOptions.miningDrillOptions) {
		return configOptions.miningDrillOptions[id], nil
	}
	return MiningDrill{}, fmt.Errorf("invalid MiningDrillID: %d", id)
}

func (configOptions *ConfigOptions) GetFurnaceID(key interface{}) (FurnaceID, error) {
	for i, furnace := range configOptions.furnaceOptions {
		if furnace.reference == key {
			return FurnaceID(i), nil
		}
	}
	return FurnaceID(-1), fmt.Errorf("no furnace registered for: %v", key)
}

func (configOptions *ConfigOptions) GetFurnaceFromID(id FurnaceID) (Furnace, error) {
	if id >= 0 && int(id) < len(configOptions.furnaceOptions) {
		return configOptions.furnaceOptions[id], nil
	}
	return Furnace{}, fmt.Errorf("invalid FurnaceID: %d", id)
}

// GetModuleFromID retrieves a Module instance by its ID.
func (configOptions *ConfigOptions) GetModuleFromID(id ModuleID) (Module, error) {
	if id >= 0 && int(id) < len(configOptions.moduleOptions) {
		return configOptions.moduleOptions[id], nil
	}
	return Module{}, fmt.Errorf("invalid ModuleID: %d", id)
}

type ProductionConfig struct {
	preferredAssemblyMachine AssemblingMachineID
	preferredMiningDrill     MiningDrillID
	preferredModules         []ModuleID

	configOptions ConfigOptions
	cache         map[string]Setup
}

// sets up production config from data schema and sets some preferred machines (will be set to -1 if nil)
func NewProductionConfigFrom(data data.Schema, amKey interface{}, mdKey interface{}, modKeys []interface{}) ProductionConfig {
	config := ProductionConfig{
		preferredModules: []ModuleID{},
	}

	config.configOptions = NewConfigOptionsFrom(data)

	if amKey != nil {
		amID, err := config.configOptions.GetAssemblingMachineID(amKey)
		if err != nil {
			log.Print(err)
		} else {
			config.preferredAssemblyMachine = amID
		}
	}

	if mdKey != nil {
		mdID, err := config.configOptions.GetMiningDrillID(mdKey)
		if err != nil {
			log.Print(err)
		} else {
			config.preferredMiningDrill = mdID
		}
	}

	for _, modKey := range modKeys {
		modID, err := config.configOptions.GetModuleID(modKey)
		if err != nil {
			log.Print(err)
		} else {
			config.preferredModules = append(config.preferredModules, modID)
		}
	}

	return config
}

type Setup struct {
	ID      interface{}
	modules []ModuleID

	EnergySource utils.EnergySource

	bonus Effect
}

func (config *ProductionConfig) GetSetupFor(recipe Recipe, strict bool) (Setup, error) {
	if config == nil {
		return Setup{}, errors.New("uninitialized config")
	}

	if config.cache == nil {
		config.cache = make(map[string]Setup)
	}

	if strict {
		log.Panic("strict setup search not implemented")
	}

	if setup, ok := config.cache[recipe.Category]; ok {
		return setup, nil
	} else {
		var res Setup

		for amID, am := range config.configOptions.assemblingMachineOptions {
			for _, crafting_category := range am.craftingCategories {
				if crafting_category == recipe.Category {
					var newRes Setup

					bonus := NewEffect()
					bonus.speed *= am.craftingSpeed

					for i := 0; i < am.moduleSlots; i++ {
						if i >= len(config.preferredModules) {
							break
						}

						res.modules = append(res.modules, config.preferredModules[i])

						mod, err := config.configOptions.GetModuleFromID(config.preferredModules[i])
						if err != nil {
							log.Fatal(err)
						}

						bonus.speed *= mod.effect.speed
						bonus.consumption *= mod.effect.consumption
						bonus.productivity *= mod.effect.productivity
					}

					newRes.bonus = bonus
					newRes.ID = AssemblingMachineID(amID)
					newRes.EnergySource = am.energySource

					if res.ID == nil {
						res = newRes
					} else if res.bonus.Value() < newRes.bonus.Value() {
						res = newRes
					}
				}
			}
		}

		for _, md := range config.configOptions.miningDrillOptions {
			for mdID, resource_category := range md.resourceCategories {
				if resource_category == recipe.Category {
					var newRes Setup

					bonus := NewEffect()
					bonus.speed *= md.miningSpeed

					for i := 0; i < md.moduleSlots; i++ {
						if i >= len(config.preferredModules) {
							break
						}

						res.modules = append(res.modules, config.preferredModules[i])

						mod, err := config.configOptions.GetModuleFromID(config.preferredModules[i])
						if err != nil {
							log.Fatal(err)
						}

						bonus.speed *= mod.effect.speed
						bonus.consumption *= mod.effect.consumption
						bonus.productivity *= mod.effect.productivity
					}

					newRes.bonus = bonus
					newRes.ID = MiningDrillID(mdID)
					newRes.EnergySource = md.energySource

					if res.ID == nil {
						res = newRes
					} else if res.bonus.Value() < newRes.bonus.Value() {
						res = newRes
					}
				}
			}
		}

		for _, f := range config.configOptions.furnaceOptions {
			for fID, resource_category := range f.craftingCategories {
				if resource_category == recipe.Category {
					var newRes Setup

					bonus := NewEffect()
					bonus.speed *= f.smeltingSpeed

					for i := 0; i < f.moduleSlots; i++ {
						if i >= len(config.preferredModules) {
							break
						}

						res.modules = append(res.modules, config.preferredModules[i])

						mod, err := config.configOptions.GetModuleFromID(config.preferredModules[i])
						if err != nil {
							log.Fatal(err)
						}

						bonus.speed *= mod.effect.speed
						bonus.consumption *= mod.effect.consumption
						bonus.productivity *= mod.effect.productivity
					}

					newRes.bonus = bonus
					newRes.EnergySource = f.energySource
					newRes.ID = FurnaceID(fID)

					if res.ID == nil {
						res = newRes
					} else if res.bonus.Value() < newRes.bonus.Value() {
						res = newRes
					}
				}
			}
		}

		if res.ID != nil {
			return res, nil
		}
	}

	return Setup{}, fmt.Errorf("no valid assembly machine found for recipe: %v", recipe)
}

func (setup Setup) GetBonus() Effect {
	return setup.bonus
}
