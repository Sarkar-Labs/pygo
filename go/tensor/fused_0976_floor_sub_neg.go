package tensor

import "fmt"

func Fused0976FloorSubNeg(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("floor_sub_neg: nil operand")
	}
	r := x.Sub(y)
	r = r.Floor()
	r = r.Neg()
	return r, nil
}
func Fused0976FloorSubNegName() string { return "floor_sub_neg" }
