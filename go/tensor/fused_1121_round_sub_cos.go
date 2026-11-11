package tensor

import "fmt"

func Fused1121RoundSubCos(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("round_sub_cos: nil operand")
	}
	r := x.Sub(y)
	r = r.Round()
	r = r.Cos()
	return r, nil
}
func Fused1121RoundSubCosName() string { return "round_sub_cos" }
