package logistic

import (
	"log"

	"github.com/factorio-ai/internal/planner/production"
	"github.com/factorio-ai/internal/utils"
)

type Orientation int

const (
	LEFT Orientation = iota
	UP
	RIGHT
	DOWN
	UNDEFINED
)

type RecipeType int

const (
	CRAFTING RecipeType = iota
	CENTRIFUGING
	SMELTING
	CHEMISTRY
	OIL_PROCESSING
	CRAFTING_WITH_FLUID
	ROCKET_BUILDING
	ADVANCED_CRAFTING
	BASIC_SOLID
	BASIC_FLUID
	WATER
)

func RecipeTyeFrom(category string) RecipeType {
	switch category {
	case "crafting":
		return CRAFTING
	case "centrifuging":
		return CENTRIFUGING
	case "semlting":
		return SMELTING
	case "chemistry":
		return CHEMISTRY
	case "oil-processing":
		return OIL_PROCESSING
	case "crafting-with-fluid":
		return CRAFTING_WITH_FLUID
	case "rocket-building":
		return ROCKET_BUILDING
	case "advanced-crafting":
		return ADVANCED_CRAFTING
	case "basic-solid":
		return BASIC_SOLID
	case "basic-fluid":
		return BASIC_FLUID
	case "water":
		return WATER
	}

	log.Printf("WARNING: crafting category %s has not been seen before", category)

	return CRAFTING
}

type AssemblingMachineNode struct {
	consumptionNeed map[production.ProductID]float64
	consumptionMet  map[production.ProductID]float64

	recipeType  RecipeType
	orientation Orientation

	energyConsumptionNeed utils.EnergySource
	energyConsumptionMet  float64

	production           map[production.ProductID]float64
	productionMultiplier float64
}

func AssemblingMachineNodeFrom(bonus production.Effect, energySource utils.EnergySource, recipe production.Recipe) AssemblingMachine {
	consumptionModifier, productivityModifier, speedModifier := bonus.Decompose()

	am := AssemblingMachineNode{
		consumptionNeed: make(map[production.ProductID]float64),
		consumptionMet:  make(map[production.ProductID]float64),

		recipeType:  RecipeTyeFrom(recipe.Category),
		orientation: UNDEFINED,

		energyConsumptionNeed: energySource,
		energyConsumptionMet:  0.0,

		production:           make(map[production.ProductID]float64),
		productionMultiplier: 1.0,
	}

	for _, input := range recipe.Input {
		am.consumptionNeed[input.ProductID] = input.Quantity * consumptionModifier * speedModifier
	}

	for _, output := range recipe.GetOutput() {
		am.production[output.ProductID] = output.Quantity * productivityModifier * speedModifier
	}

	return am
}
