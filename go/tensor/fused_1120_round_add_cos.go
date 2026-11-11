package tensor

import "fmt"

func Fused1120RoundAddCos(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("round_add_cos: nil operand")
	}
	r := x.Add(y)
	r = r.Round()
	r = r.Cos()
	return r, nil
}
func Fused1120RoundAddCosName() string { return "round_add_cos" }
