package tensor

import "fmt"

func Fused0313TanhDivCos(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("tanh_div_cos: nil operand")
	}
	r := x.Div(y)
	r = r.Tanh()
	r = r.Cos()
	return r, nil
}
func Fused0313TanhDivCosName() string { return "tanh_div_cos" }
