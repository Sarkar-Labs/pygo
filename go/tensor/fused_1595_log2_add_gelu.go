package tensor

import "fmt"

func Fused1595Log2AddGelu(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("log2_add_gelu: nil operand")
	}
	r := x.Add(y)
	r = r.Log2()
	r = r.GELU()
	return r, nil
}
func Fused1595Log2AddGeluName() string { return "log2_add_gelu" }
