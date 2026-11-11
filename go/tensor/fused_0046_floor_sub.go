package tensor

import "fmt"

func Fused0046FloorSub(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("floor_sub: nil operand")
	}
	r := x.Sub(y)
	r = r.Floor()
	return r, nil
}
func Fused0046FloorSubName() string { return "floor_sub" }
