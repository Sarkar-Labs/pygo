package tensor

import "fmt"

func Fused0013TanhDiv(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("tanh_div: nil operand")
	}
	r := x.Div(y)
	r = r.Tanh()
	return r, nil
}
func Fused0013TanhDivName() string { return "tanh_div" }
