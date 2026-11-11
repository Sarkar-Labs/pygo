package tensor

import "fmt"

func Fused0920FloorAddSqrt(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("floor_add_sqrt: nil operand")
	}
	r := x.Add(y)
	r = r.Floor()
	r = r.Sqrt()
	return r, nil
}
func Fused0920FloorAddSqrtName() string { return "floor_add_sqrt" }
