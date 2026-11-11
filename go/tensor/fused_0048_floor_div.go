package tensor

import "fmt"

func Fused0048FloorDiv(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("floor_div: nil operand")
	}
	r := x.Div(y)
	r = r.Floor()
	return r, nil
}
func Fused0048FloorDivName() string { return "floor_div" }
