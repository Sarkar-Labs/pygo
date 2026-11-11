package tensor

import "fmt"

func Fused0088Log10Div(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("log10_div: nil operand")
	}
	r := x.Div(y)
	r = r.Log10()
	return r, nil
}
func Fused0088Log10DivName() string { return "log10_div" }
