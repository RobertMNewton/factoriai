package main

import (
	"log"

	"github.com/factorio-ai/internal/data"
	"github.com/factorio-ai/internal/planner/production"
)

func main() {
	pm := production.NewProductManagerFrom(data.Data)
	rm := production.NewRecipeManagerFrom(data.Data, &pm)

	var problem map[string]float64 = map[string]float64{"automation-science-pack": 1, "logistic-science-pack": 1}

	config := production.NewProductionConfigFrom(data.Data, nil, nil, nil)
	productionGraph := production.NewProductionSolver(&pm, &rm, &config, nil)
	for name, quantity := range problem {
		productionGraph.AddProblem(
			pm.NewProductionFrom(name, quantity),
		)
	}

	productionGraph.Solve(false)
	productionGraph.DisplaySolution()

	recipes := rm.GetRecipes()
	unique_categories := make([]string, 0, 10)
	for _, recipe := range recipes {
		found := false
		for _, category := range unique_categories {
			if category == recipe.Category {
				found = true
				break
			}
		}

		if !found {
			unique_categories = append(unique_categories, recipe.Category)
		}
	}

	log.Println(unique_categories)
}
