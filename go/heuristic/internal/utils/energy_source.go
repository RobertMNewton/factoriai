package utils

type EnergySourceType int

const (
	COMBUSTION EnergySourceType = iota
	ELECTRIC
)

type EnergySource struct {
	EnergyType   EnergySourceType
	EnergyAmount float64
}
