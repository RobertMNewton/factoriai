package logistic

import "github.com/factorio-ai/internal/planner/production"

type LogisticNode interface {
	GetNodes() ([]*LogisticNode, []*LogisticNode) // (in-going nodes, out-going nodes
	AddOutNode(node *LogisticNode, info interface{}) error
	AddInNode(node *LogisticNode, info interface{}) error
	Compute() error                         // compute that the nodes input requirements are met and set internal fields to hold relevent data
	GetProduction() []production.Production // gets potential outputs.
}

type LogisticGraph struct{}
