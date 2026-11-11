package tensor

import "fmt"

func Fused1571Log2SubCos(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("log2_sub_cos: nil operand")
	}
	r := x.Sub(y)
	r = r.Log2()
	r = r.Cos()
	return r, nil
}
func Fused1571Log2SubCosName() string { return "log2_sub_cos" }
