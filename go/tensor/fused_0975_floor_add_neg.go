package tensor

import "fmt"

func Fused0975FloorAddNeg(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("floor_add_neg: nil operand")
	}
	r := x.Add(y)
	r = r.Floor()
	r = r.Neg()
	return r, nil
}
func Fused0975FloorAddNegName() string { return "floor_add_neg" }
