package logistic

type BeltEdge struct {
	isInput bool
	node    *LogisticNode
}

// This is an abstraction of a transport belt. View it as going left-to-right (obj)
// with nodes coming in and out from the top and bottom respectively
type Belt struct {
	inNode  *LogisticNode
	outNode *LogisticNode

	topNodes    []*BeltEdge
	bottomNodes []*BeltEdge
}
