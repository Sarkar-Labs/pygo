package tensor

import "fmt"

func Fused1570Log2AddCos(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("log2_add_cos: nil operand")
	}
	r := x.Add(y)
	r = r.Log2()
	r = r.Cos()
	return r, nil
}
func Fused1570Log2AddCosName() string { return "log2_add_cos" }
