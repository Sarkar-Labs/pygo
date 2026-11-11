package tensor

import "fmt"

func Fused1565Log2AddSin(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("log2_add_sin: nil operand")
	}
	r := x.Add(y)
	r = r.Log2()
	r = r.Sin()
	return r, nil
}
func Fused1565Log2AddSinName() string { return "log2_add_sin" }
