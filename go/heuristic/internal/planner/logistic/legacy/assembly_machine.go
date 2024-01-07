package logistic

import (
	"fmt"
	"log"

	"github.com/factorio-ai/internal/planner/production"
)

type AssemblyMachine struct {
	inNodes  []*LogisticNode
	outNodes []*LogisticNode

	recipeID production.RecipeID
	setup    production.Setup

	production []production.Production
}

func NewAssemblyMachine(recipe production.RecipeID, setup production.Setup) AssemblyMachine {
	var am AssemblyMachine
	am.recipeID = recipe
	am.setup = setup

	return am
}

func (am AssemblyMachine) GetNodes() ([]*LogisticNode, []*LogisticNode) {
	return am.inNodes, am.outNodes
}

func (am *AssemblyMachine) AddOutNode(node *LogisticNode, info interface{}) error {
	if len(am.inNodes)+len(am.outNodes) > 9 {
		return fmt.Errorf("could append out-node %v to assembly machine due to maximum node connections being reached", *node)
	}
	am.outNodes = append(am.outNodes, node)
	return nil
}

func (am *AssemblyMachine) AddInNode(node *LogisticNode, info interface{}) error {
	if len(am.inNodes)+len(am.outNodes) > 9 {
		return fmt.Errorf("could append in-node %v to assembly machine due to maximum node connections being reached", *node)
	}
	am.inNodes = append(am.outNodes, node)
	return nil
}

func (am *AssemblyMachine) Compute(rm *production.RecipeManager, pm *production.ProductManager) error {
	recipe := rm.GetRecipe(am.recipeID)

	inputs := make(map[production.ProductID]float64)
	for _, inNode := range am.inNodes {
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

	am.production = make([]production.Production, 0, len(recipe.Output))
	for _, output := range recipe.Output {
		output.Quantity *= slowestProductionSpeed
		am.production = append(am.production, output)
	}

	return nil
}

func (am AssemblyMachine) GetProduction() []production.Production {
	return am.production
}
