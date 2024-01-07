package utils

type ListNode struct {
	value interface{}
	front *ListNode
}

func (node *ListNode) Value() interface{} {
	return node.value
}

type LinkedList struct {
	front *ListNode
	back  *ListNode
	len   int
}

func NewLinkedList() LinkedList {
	return LinkedList{}
}

func (ll *LinkedList) Append(value interface{}) {
	new_node := ListNode{
		value: value,
	}

	if ll.len == 0 {
		ll.front = &new_node
		ll.back = &new_node
	} else {
		ll.front.front = &new_node
		ll.front = &new_node
	}

	ll.len++
}

func (ll *LinkedList) Pop() *ListNode {
	if ll.len > 0 {
		popped_node := ll.back
		ll.back = popped_node.front

		ll.len--

		return popped_node
	} else {
		return nil
	}
}

func (ll *LinkedList) Size() int {
	return ll.len
}
