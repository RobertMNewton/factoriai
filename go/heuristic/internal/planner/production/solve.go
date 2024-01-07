package production

import (
	"fmt"
	"log"
	"strings"

	"github.com/factorio-ai/internal/utils"
)

type Solution struct {
	Amount float64
	Setup  Setup
}

type ProductionSolver struct {
	pm     *ProductManager
	rm     *RecipeManager
	config *ProductionConfig

	solution map[RecipeID]Solution
	problem  []Production
}

func NewProductionSolver(pm *ProductManager, rm *RecipeManager, config *ProductionConfig, problem []Production) (pg ProductionSolver) {
	pg.pm = pm
	pg.rm = rm
	pg.config = config
	pg.solution = make(map[RecipeID]Solution)

	if problem != nil {
		pg.problem = problem
	} else {
		pg.problem = make([]Production, 0)
	}

	return pg
}

func (pg *ProductionSolver) AddProblem(problem Production) {
	pg.problem = append(pg.problem, problem)
}

func (pg *ProductionSolver) Solve(strictConfig bool) {
	// inital set up
	problems := utils.NewLinkedList()
	for _, production := range pg.problem {
		problems.Append(production)
	}

	for node := problems.Pop(); node != nil; node = problems.Pop() {
		cp := node.Value().(Production)

		valid_recipes := pg.rm.GetCatalogProducing(cp.ProductID)
		if len(valid_recipes) == 1 {
			setup, err := pg.config.GetSetupFor(pg.rm.GetRecipe(valid_recipes[0].RecipeID), strictConfig)
			if err != nil {
				log.Fatalf("Error: %e", err)
			}

			recipeID := valid_recipes[0].RecipeID
			product_qty := valid_recipes[0].float64 * setup.bonus.productivity * setup.bonus.speed

			recipe_qty := cp.Quantity / product_qty

			if solution, ok := pg.solution[recipeID]; ok {
				solution.Amount += recipe_qty
				pg.solution[recipeID] = solution
			} else {
				pg.solution[recipeID] = Solution{
					Amount: recipe_qty,
					Setup:  setup,
				}
			}

			for _, input := range pg.rm.GetRecipe(recipeID).Input {
				input.Quantity *= recipe_qty * setup.bonus.consumption
				problems.Append(input)
			}
		} else if len(valid_recipes) > 1 {
			product, _ := pg.pm.GetProduct(cp.ProductID)
			var recipeID RecipeID
			var qty float64

			var _ interface{}
			if product == "petroleum-gas" {
				recipeID, _ = pg.rm.GetRecipeID("basic-oil-processing")
				qty, _ = pg.rm.GetRecipe(recipeID).GetOutputFor(cp.ProductID)
			} else if product == "light-oil" || product == "heavy-oil" {
				recipeID, _ = pg.rm.GetRecipeID("advanced-oil-processing")
				qty, _ = pg.rm.GetRecipe(recipeID).GetOutputFor(cp.ProductID)
			} else {
				log.Fatalf("no support for product: %v", cp.ProductID)
			}

			log.Print("Warning: oil Processing Recipes are not optimized!")

			setup, err := pg.config.GetSetupFor(pg.rm.GetRecipe(recipeID), strictConfig)
			if err != nil {
				log.Fatalf("Error: %e", err)
			}

			product_qty := qty * setup.bonus.productivity * setup.bonus.speed

			recipe_qty := cp.Quantity / product_qty

			if solution, ok := pg.solution[recipeID]; ok {
				solution.Amount += recipe_qty
				pg.solution[recipeID] = solution
			} else {
				pg.solution[recipeID] = Solution{
					Amount: recipe_qty,
					Setup:  setup,
				}
			}

			for _, input := range pg.rm.GetRecipe(recipeID).Input {
				input.Quantity *= recipe_qty * setup.bonus.consumption
				problems.Append(input)
			}
		} else {
			product, err := pg.pm.GetProduct(cp.ProductID)
			if err != nil {
				panic(err)
			} else {
				panic(fmt.Sprintf("No recipes found for product: %v", product))
			}
		}
	}
}

func (pg *ProductionSolver) DisplaySolution() {
	// Print the header of the table
	fmt.Printf("%-20s %-10s %-20s %-10s\n", "Recipe", "Setup", "Setup ID", "Quantity")
	fmt.Println(strings.Repeat("-", 60)) // Print a separator line

	for recipeID, solution := range pg.solution {
		recipe := pg.rm.GetRecipe(recipeID)

		recipeName, ok := recipe.Reference.(string)
		if !ok {
			recipeName = "<nil>"
		}

		setupID := "<nil>"
		if solution.Setup.ID != nil {
			idType := "<nil>"
			switch solution.Setup.ID.(type) {
			case AssemblingMachineID:
				idType = "AssemblyMachineID"
			case FurnaceID:
				idType = "FurnaceID"
			case MiningDrillID:
				idType = "MiningDrillID"
			}

			setupID = fmt.Sprintf("%v=%v", idType, solution.Setup.ID)
		}

		// Print each row of the table
		fmt.Printf("%-20s %-10d %-20s %-10f\n", recipeName, recipeID, setupID, solution.Amount)
	}
}
