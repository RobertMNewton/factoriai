package logistic

import (
	"fmt"
	"log"

	"github.com/factorio-ai/internal/planner/production"
)

type MiningMachine struct {
	inNodes  []*LogisticNode
	outNodes []*LogisticNode

	recipeID production.RecipeID
	setup    production.Setup

	production []production.Production
}

func NewMiningMachine(recipe production.RecipeID, setup production.Setup) MiningMachine {
	var mm MiningMachine
	mm.recipeID = recipe
	mm.setup = setup

	return mm
}

func (mm MiningMachine) GetNodes() ([]*LogisticNode, []*LogisticNode) {
	return mm.inNodes, mm.outNodes
}

func (mm *MiningMachine) AddOutNode(node *LogisticNode) error {
	if len(mm.inNodes)+len(mm.outNodes) > 9 {
		return fmt.Errorf("could append out-node %v to assembly machine due to maximum node connections being reached", *node)
	}
	mm.outNodes = append(mm.outNodes, node)
	return nil
}

func (mm *MiningMachine) AddInNode(node *LogisticNode) error {
	if len(mm.inNodes)+len(mm.outNodes) > 9 {
		return fmt.Errorf("could append in-node %v to assembly machine due to maximum node connections being reached", *node)
	}
	mm.inNodes = append(mm.outNodes, node)
	return nil
}

func (mm *MiningMachine) Compute(rm *production.RecipeManager, pm *production.ProductManager) error {
	recipe := rm.GetRecipe(mm.recipeID)

	inputs := make(map[production.ProductID]float64)
	for _, inNode := range mm.inNodes {
		productions := (*inNode).GetProduction()
		for _, production := range productions {
			inputs[production.ProductID] = production.Quantity
		}
	}

	var slowestProductionSpeed float64 = 1
	for _, inputCheck := range recipe.Input {
		if inputQty, ok := inputs[inputCheck.ProductID]; ok {
			productionSpeed := inputCheck.Quantity / inputQty
			if productionSpeed < slowestProductionSpeed {
				slowestProductionSpeed = productionSpeed
			}
		} else {
			product, err := pm.GetProduct(inputCheck.ProductID)
			if err != nil {
				log.Panicf("Error: %e", err)
			}
			return fmt.Errorf("assemebly machine is missing required input: %v", product)
		}
	}

	mm.production = make([]production.Production, 0, len(recipe.Output))
	for _, output := range recipe.Output {
		output.Quantity *= slowestProductionSpeed
		mm.production = append(mm.production, output)
	}

	return nil
}

func (mm MiningMachine) GetProduction() []production.Production {
	return mm.production
}
