package tensor

import "fmt"

func Fused0015AbsAdd(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("abs_add: nil operand")
	}
	r := x.Add(y)
	r = r.Abs()
	return r, nil
}
func Fused0015AbsAddName() string { return "abs_add" }
